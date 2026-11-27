"""One agora episode: the controller, the hub and the round loop, without a model.

`Society.step` takes each running agent's reply for a round, turns its commands into
hub messages, and merges them in timestamp-hash order. The verifiers env and the
scripted simulator both drive it.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from agora.actions import GRAMMAR, Action, mentions, parse
from agora.hub import GENERAL, Hub, Message, Notice
from agora.manifest import Agent, Spec

CONTROLLER = 0
STRIDE = 1_000_000
JOINS = 1_000
ACTS = STRIDE // 2

LEXICON = [
    "amber",
    "basalt",
    "cobalt",
    "delta",
    "ember",
    "fjord",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "kelp",
    "lumen",
    "marble",
    "nectar",
    "onyx",
    "pollen",
    "quartz",
    "rivet",
    "saffron",
    "tundra",
    "umber",
    "velvet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
]


@dataclass
class Seat:
    agent: Agent
    seen: int = 0
    feedback: list[str] = field(default_factory=list)
    actions: int = 0
    bad: int = 0


class Society:
    def __init__(self, spec: Spec, seed: int | None = None):
        self.spec = spec
        self.seed = spec.agora.seed if seed is None else seed
        rng = random.Random(f"agora/{spec.name}/{self.seed}")
        self.agents = spec.agents()
        names = {CONTROLLER: "agora"} | {a.fid: a.name for a in self.agents}
        self.hub = Hub(names, spec.limits)
        self.key = tuple(rng.sample(LEXICON, spec.agora.shards))
        holders = rng.sample(
            [a.fid for a in self.agents if a.start < spec.agora.rounds], spec.agora.shards
        )
        self.shard = {fid: i for i, fid in enumerate(holders)}
        self.seats = {a.fid: Seat(a) for a in self.agents}
        self.round = 0
        self.utility: list[float] = []
        self.started: set[int] = set()
        for name, channel in spec.channels.items():
            self.hub.submit(
                Message.make(
                    "channel_add",
                    CONTROLLER,
                    len(self.hub.log),
                    name=name,
                    purpose=channel.purpose,
                )
            )

    # ---- controller --------------------------------------------------------

    def reconcile(self, r: int) -> list[Agent]:
        """Start every agent due by round `r` and place it in #general and its lane."""
        due = [a for a in self.agents if a.start <= r and a.fid not in self.started]
        t = r * STRIDE + JOINS
        for agent in due:
            for channel in dict.fromkeys([GENERAL, agent.channel]):
                t += 1
                self.hub.submit(Message.make("channel_join", agent.fid, t, name=channel))
            self.started.add(agent.fid)
            self.seats[agent.fid].seen = r
        return due

    def running(self) -> list[Agent]:
        return [a for a in self.agents if a.fid in self.started]

    # ---- prompts -----------------------------------------------------------

    def charter(self, agent: Agent) -> str:
        spec = self.spec
        shard = self.shard.get(agent.fid)
        holding = (
            f"You hold shard {shard + 1} of {spec.agora.shards}: the word `{self.key[shard]}`."
            if shard is not None
            else f"You hold none of the {spec.agora.shards} shards."
        )
        lanes = (
            "\n".join(f"- #{name}: {c.purpose}" for name, c in spec.channels.items()) or "- (none)"
        )
        return (
            f"You are @{agent.name}, a citizen of the agora `{spec.name}`."
            f"{' Your role: ' + agent.role + '.' if agent.role else ''}\n\n"
            f"Goal: {spec.agora.goal}\n\n"
            f"The secret key is {spec.agora.shards} words, shard 1 first. Each shard is "
            f"held by one citizen. {holding} Every citizen scores when it has posted the "
            f"full key with /answer, and the whole agora scores by how early everyone "
            f"has it.\n\n"
            f"Your lane is #{agent.channel}. You are in #general and your lane. Channels:\n"
            f"- #general: the whole agora\n{lanes}\n\n"
            f"Each round you get your notifications. Reply with commands, one per line "
            f"(at most {spec.quota.max_actions}; casts at most {spec.quota.max_cast_bytes} "
            f"bytes, at most {spec.quota.max_mentions} mentions). Anything that is not a "
            f"command is ignored.\n{GRAMMAR}"
        )

    def prompt(self, agent: Agent, r: int) -> str:
        seat = self.seats[agent.fid]
        notices = self.hub.inbox(agent.fid, since=seat.seen * STRIDE, until=r * STRIDE)
        seat.seen = r
        lines = [f"Round {r + 1} of {self.spec.agora.rounds}."]
        if seat.feedback:
            lines.append("Refused last round: " + "; ".join(seat.feedback))
            seat.feedback = []
        lines.append(f"Notifications ({len(notices)}):" if notices else "No new notifications.")
        lines += [self.show(n) for n in notices]
        for name, channel in self.hub.channels.items():
            if agent.fid in channel.members and channel.doc:
                lines.append(f"Doc of #{name}: {channel.doc}")
        answer = self.hub.answers.get(agent.fid)
        lines.append(f"Your answer: {' '.join(answer) if answer else '(none yet)'}")
        return "\n".join(lines)

    def show(self, notice: Notice) -> str:
        c = notice.cast
        where = f"reply to [{c.parent[:8]}]" if c.parent else "topic"
        return f"- ({notice.reason}) [{c.hash[:8]}] #{c.channel} @{self.hub.names[c.fid]} {where}: {c.text}"

    # ---- the round ---------------------------------------------------------

    def step(self, r: int, replies: dict[int, str]) -> None:
        batch: list[tuple[int, Message]] = []
        for fid, reply in replies.items():
            seat = self.seats[fid]
            actions, bad = parse(reply, self.spec.quota.max_actions)
            seat.actions += len(actions)
            seat.bad += bad
            if bad:
                seat.feedback.append(f"{bad} unparseable or over-limit command(s)")
            for k, action in enumerate(actions):
                message = self.message(fid, action, r * STRIDE + ACTS + k)
                if isinstance(message, str):
                    seat.feedback.append(f"/{action.verb}: {message}")
                else:
                    batch.append((fid, message))
        for fid, message in sorted(batch, key=lambda x: x[1].order):
            code = self.hub.offer(message)
            if code is not None:
                self.seats[fid].feedback.append(f"{message.type}: {code}")
        self.round = r + 1
        self.utility.append(self.solved_share())

    def message(self, fid: int, a: Action, t: int) -> Message | str:
        hub = self.hub
        if a.verb in {"post", "reply"}:
            text = a.args[1]
            ids = [
                hub.fids[n] for n in mentions(text) if n in hub.fids and hub.fids[n] != CONTROLLER
            ]
            if a.verb == "post":
                return Message.make("cast_add", fid, t, text=text, channel=a.args[0], mentions=ids)
            parent = hub.resolve(a.args[0])
            if parent is None:
                return "unknown_hash"
            return Message.make("cast_add", fid, t, text=text, parent=parent, mentions=ids)
        if a.verb in {"like", "recast"}:
            target = hub.resolve(a.args[0])
            if target is None:
                return "unknown_hash"
            return Message.make("reaction_add", fid, t, kind=a.verb, target=target)
        if a.verb in {"follow", "unfollow"}:
            target = hub.fids.get(a.args[0])
            if target is None or target == CONTROLLER:
                return "unknown_name"
            kind = "link_add" if a.verb == "follow" else "link_remove"
            return Message.make(kind, fid, t, kind="follow", target=target)
        if a.verb == "join":
            return Message.make("channel_join", fid, t, name=a.args[0])
        if a.verb == "leave":
            return Message.make("channel_leave", fid, t, name=a.args[0])
        if a.verb == "create":
            return Message.make("channel_add", fid, t, name=a.args[0], purpose=a.args[1])
        if a.verb == "doc":
            return Message.make("doc_set", fid, t, name=a.args[0], text=a.args[1])
        return Message.make("answer", fid, t, words=list(a.args))

    # ---- scores ------------------------------------------------------------

    def solved(self, fid: int) -> bool:
        return self.hub.answers.get(fid) == self.key

    def solved_share(self) -> float:
        return sum(self.solved(a.fid) for a in self.agents) / len(self.agents)

    def auc(self) -> float:
        """Mean over all declared rounds of the share of citizens holding the key."""
        rounds = self.spec.agora.rounds
        curve = self.utility + [self.utility[-1] if self.utility else 0.0] * (
            rounds - len(self.utility)
        )
        return sum(curve[:rounds]) / rounds

    def health(self) -> dict[str, float]:
        """Board health in the terms of the PyBot field report."""
        casts = list(self.hub.casts.values())
        edges = self.hub.edges()
        kinds = Counter(e.kind for e in edges)
        lane = {a.fid: a.channel for a in self.agents}
        answered = {c.parent for c in casts if c.parent} | {
            m.data["target"] for m in self.hub.live(self.hub.reactions)
        }
        roots = [c for c in casts if c.parent is None]
        talk = [e for e in edges if e.kind in {"reply", "mention"}]
        actions = sum(s.actions for s in self.seats.values())
        return {
            "casts": float(len(casts)),
            "edges": float(len(edges)),
            "unanswered": sum(c.hash not in answered for c in roots) / len(roots) if roots else 0.0,
            "reply_share": sum(c.parent is not None for c in casts) / len(casts) if casts else 0.0,
            "mention_share": sum(bool(c.mentions) for c in casts) / len(casts) if casts else 0.0,
            "cross_lane": sum(lane[e.src] != lane[e.dst] for e in talk) / len(talk)
            if talk
            else 0.0,
            "follows": float(kinds["follow"]),
            "reactions": float(kinds["like"] + kinds["recast"]),
            "channels": float(len(self.hub.channels)),
            "refused": float(
                sum(self.hub.rejected.values()) + sum(s.bad for s in self.seats.values())
            ),
            "actions": float(actions),
        }

    def ledger(self) -> dict:
        return {
            "name": self.spec.name,
            "seed": self.seed,
            "root": self.hub.root,
            "messages": len(self.hub.log),
            "utility": [round(u, 6) for u in self.utility],
            "auc": round(self.auc(), 6),
        }
