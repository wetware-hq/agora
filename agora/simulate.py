"""Run an agora with scripted citizens instead of a model: a load test for the hub
and controller, and a baseline for the board's health metrics.

    uv run agora-sim --manifest polis --seed 0
"""

from __future__ import annotations

import argparse
import json
import re

from agora.manifest import load
from agora.society import Society
from agora.taskset import manifest_docs

SHARD = re.compile(r"shard (\d+): ([a-z]+)")


def gossip(society: Society, fid: int, prompt: str, known: dict[int, str]) -> str:
    """Post your shard to #general, pin what you know to your lane's doc (and the full
    key to #general's, for late joiners), and answer once complete."""
    for i, word in SHARD.findall(prompt):
        known[int(i)] = word
    agent = society.seats[fid].agent
    lines: list[str] = []
    shard = society.shard.get(fid)
    if shard is not None and shard + 1 not in known:
        known[shard + 1] = society.key[shard]
        lines.append(f"/post #general shard {shard + 1}: {society.key[shard]}")
    n = society.spec.agora.shards
    if len(known) == n:
        lines.append("/answer " + " ".join(known[i] for i in range(1, n + 1)))
        if "Doc of #general" not in prompt:
            lines.append(
                "/doc #general " + "; ".join(f"shard {i}: {w}" for i, w in sorted(known.items()))
            )
    if known and agent.channel != "general":
        lines.append(
            f"/doc #{agent.channel} "
            + "; ".join(f"shard {i}: {w}" for i, w in sorted(known.items()))
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="agora")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    society = Society(load(manifest_docs(args.manifest)), args.seed)
    memory: dict[int, dict[int, str]] = {}
    for r in range(society.spec.agora.rounds):
        society.reconcile(r)
        replies = {
            a.fid: gossip(society, a.fid, society.prompt(a, r), memory.setdefault(a.fid, {}))
            for a in society.running()
        }
        society.step(r, replies)
    print(json.dumps(society.ledger() | {"health": society.health()}, indent=2))


if __name__ == "__main__":
    main()
