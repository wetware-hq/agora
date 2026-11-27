"""agora — a verifiers environment for a society of agents on one message board.

One episode is one agora: the env reads the task's manifest, runs the controller
each round (staggered waves start, notifications are pushed), opens one live
interaction per citizen, and merges every citizen's commands into the hub. Each
citizen is scored by its own final answer (`solved`) and by what the agora achieved
together (`team_auc`, the area under the share-of-citizens-solved curve).
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from importlib.resources import files

import verifiers.v1 as vf
from pydantic import Field

from agora.manifest import load
from agora.society import Society


class AgoraData(vf.TaskData):
    manifest: list[dict]
    seed: int


class AgoraTask(vf.Task[AgoraData]):
    """The episode seed; citizens are judged cross-agent by the env."""


class AgoraConfig(vf.TasksetConfig):
    manifest: str = "agora"
    """A bundled manifest name (`agora`, `polis`) or a path to a manifest YAML."""
    num_tasks: int = Field(64, ge=1)
    """Episodes, one per seed: each seed redraws the key and its holders."""


class AgoraTaskset(vf.Taskset[AgoraTask, AgoraConfig]):
    def load(self) -> list[AgoraTask]:
        docs = manifest_docs(self.config.manifest)
        spec = load(docs)
        return [
            AgoraTask(
                AgoraData(
                    name=f"{spec.name}#{seed}",
                    prompt=None,
                    manifest=docs,
                    seed=spec.agora.seed + seed,
                ),
                self.config.task,
            )
            for seed in range(self.config.num_tasks)
        ]


class AgoraEnvConfig(vf.EnvConfig):
    citizen: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})


class AgoraEnv(vf.Env[AgoraEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        society = Society(load(task.data.manifest), task.data.seed)
        seats: dict[int, vf.Interaction] = {}
        done: set[int] = set()

        async def act(fid: int, prompt: str) -> tuple[int, str]:
            segment = await seats[fid].turn(prompt)
            if segment.terminated:
                done.add(fid)
                return fid, ""
            return fid, segment.last_reply

        async with AsyncExitStack() as stack:
            for r in range(society.spec.agora.rounds):
                for agent in society.reconcile(r):
                    seat = vf.Task(
                        vf.TaskData(
                            idx=task.data.idx,
                            name=f"{task.data.name}/{agent.name}",
                            prompt=None,
                            system_prompt=society.charter(agent),
                        )
                    )
                    seats[agent.fid] = await stack.enter_async_context(
                        agents.citizen.interaction(seat)
                    )
                live = [a for a in society.running() if a.fid not in done]
                replies = await asyncio.gather(*(act(a.fid, society.prompt(a, r)) for a in live))
                society.step(r, dict(replies))
        ledger = society.ledger() | {
            "health": society.health(),
            "key": list(society.key),
        }
        for fid, seat in seats.items():
            trace = seat.trace
            seat_info = society.seats[fid]
            trace.info["agora"] = ledger | {
                "fid": fid,
                "name": seat_info.agent.name,
                "shard": society.shard.get(fid),
                "answer": list(society.hub.answers.get(fid, ())),
            }
            if trace.is_format_violation:
                continue
            trace.record_reward("solved", float(society.solved(fid)))
            trace.record_reward("team_auc", society.auc())
            trace.record_metric("actions", float(seat_info.actions))
            trace.record_metric("refused", float(seat_info.bad))
            for key, value in society.health().items():
                trace.record_metric(f"board/{key}", value)


def manifest_docs(name: str) -> list[dict]:
    from pathlib import Path

    import yaml

    path = Path(name)
    text = (
        path.read_text()
        if path.suffix in {".yaml", ".yml"}
        else files("agora").joinpath("manifests", f"{name}.yaml").read_text()
    )
    return [d for d in yaml.safe_load_all(text) if d]
