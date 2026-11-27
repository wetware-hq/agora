"""The hub: a Farcaster-style message-graph whose state is a directed multigraph.

Every change is a signed-by-fid `Message` (a delta). Messages are totally ordered by
(timestamp, hash), merged into per-type CRDTs with last-write-wins and remove-wins
rules, and appended to a hash-chained log whose root replays the whole board.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field

ROOT_EMPTY = hashlib.sha256(b"agora-hub/1").hexdigest()
GENERAL = "general"

ADDS = {"cast_add", "reaction_add", "link_add"}
REMOVES = {"cast_remove", "reaction_remove", "link_remove"}
BOARD = {"channel_add", "channel_join", "channel_leave", "doc_set", "answer"}
TYPES = ADDS | REMOVES | BOARD
REACTIONS = {"like", "recast"}
LINKS = {"follow"}


class Rejected(ValueError):
    """An admission refusal: the message is not merged and costs nothing."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class Limits:
    max_cast_bytes: int = 1200
    max_mentions: int = 10
    max_channels: int = 40
    max_doc_bytes: int = 2000
    inbox: int = 20


@dataclass(frozen=True)
class Message:
    type: str
    fid: int
    timestamp: int
    body: tuple[tuple[str, object], ...]
    hash: str

    @classmethod
    def make(cls, type: str, fid: int, timestamp: int, **body: object) -> Message:
        if type not in TYPES:
            raise Rejected("unknown_type", type)
        items = tuple(sorted((k, _freeze(v)) for k, v in body.items()))
        data = {"type": type, "fid": fid, "timestamp": timestamp, "body": dict(items)}
        digest = hashlib.sha256(_canonical(data).encode()).hexdigest()[:40]
        return cls(type, fid, timestamp, items, digest)

    @property
    def data(self) -> dict:
        return dict(self.body)

    @property
    def order(self) -> tuple[int, str]:
        return (self.timestamp, self.hash)

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "fid": self.fid,
            "timestamp": self.timestamp,
            "body": {k: list(v) if isinstance(v, tuple) else v for k, v in self.body},
            "hash": self.hash,
        }

    @classmethod
    def from_json(cls, row: dict) -> Message:
        message = cls.make(row["type"], row["fid"], row["timestamp"], **row["body"])
        if message.hash != row["hash"]:
            raise Rejected("bad_hash", row["hash"])
        return message


@dataclass
class Cast:
    hash: str
    fid: int
    text: str
    channel: str
    timestamp: int
    mentions: tuple[int, ...] = ()
    parent: str | None = None
    root: str = ""


@dataclass
class Channel:
    name: str
    purpose: str = ""
    owner: int = 0
    doc: str = ""
    doc_order: tuple[int, str] = (-1, "")
    members: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    kind: str
    hash: str


@dataclass(frozen=True)
class Notice:
    reason: str
    cast: Cast
    at: int
    """When the event behind the notice happened: the cast, or its recast."""


class Hub:
    def __init__(self, names: dict[int, str], limits: Limits = Limits()):
        self.names = dict(names)
        self.fids = {name: fid for fid, name in names.items()}
        self.limits = limits
        self.casts: dict[str, Cast] = {}
        self.tombstones: set[str] = set()
        self.reactions: dict[tuple[int, str, str], Message] = {}
        self.links: dict[tuple[int, str, int], Message] = {}
        self.channels: dict[str, Channel] = {GENERAL: Channel(GENERAL, "the whole agora")}
        self.answers: dict[int, tuple[str, ...]] = {}
        self.threads: dict[str, set[int]] = defaultdict(set)
        self.log: list[Message] = []
        self.root = ROOT_EMPTY
        self.rejected: dict[str, int] = defaultdict(int)

    # ---- admission and merge -------------------------------------------------

    def submit(self, message: Message) -> None:
        """Validate and merge one message, or raise `Rejected` with a reason code."""
        if message.fid not in self.names:
            raise Rejected("unknown_fid", str(message.fid))
        if self.log and message.order <= self.log[-1].order:
            raise Rejected("out_of_order", message.hash)
        getattr(self, f"_{message.type}")(message, message.data)
        self.log.append(message)
        self.root = hashlib.sha256(f"{self.root}{message.hash}".encode()).hexdigest()

    def offer(self, message: Message) -> str | None:
        """`submit`, returning the rejection code instead of raising."""
        try:
            self.submit(message)
        except Rejected as e:
            self.rejected[e.code] += 1
            return e.code
        return None

    def _cast_add(self, m: Message, b: dict) -> None:
        text = str(b.get("text", ""))
        if not text.strip():
            raise Rejected("empty_cast")
        if len(text.encode()) > self.limits.max_cast_bytes:
            raise Rejected("cast_too_long", f"{len(text.encode())}>{self.limits.max_cast_bytes}")
        mentions = tuple(b.get("mentions", ()))
        if len(mentions) > self.limits.max_mentions:
            raise Rejected("too_many_mentions", str(len(mentions)))
        if any(f not in self.names for f in mentions):
            raise Rejected("unknown_mention")
        parent = b.get("parent")
        if parent is not None:
            if parent not in self.casts:
                raise Rejected("unknown_parent", str(parent))
            up = self.casts[parent]
            channel, root = up.channel, up.root
        else:
            channel, root = str(b.get("channel", GENERAL)), m.hash
        if channel not in self.channels:
            raise Rejected("unknown_channel", channel)
        if m.fid not in self.channels[channel].members:
            raise Rejected("not_a_member", channel)
        if m.hash in self.tombstones:
            raise Rejected("removed", m.hash)
        cast = Cast(m.hash, m.fid, text, channel, m.timestamp, mentions, parent, root)
        self.casts[m.hash] = cast
        self.threads[root].add(m.fid)

    def _cast_remove(self, m: Message, b: dict) -> None:
        target = b.get("target")
        cast = self.casts.get(target)
        if cast is None or cast.fid != m.fid:
            raise Rejected("not_your_cast", str(target))
        self.tombstones.add(target)
        del self.casts[target]

    def _lww(self, table: dict, key: tuple, m: Message) -> None:
        old = table.get(key)
        # Farcaster: the higher timestamp wins; on a tie a remove beats an add,
        # then the higher hash wins.
        if old is not None and (old.timestamp, old.type in REMOVES, old.hash) >= (
            m.timestamp,
            m.type in REMOVES,
            m.hash,
        ):
            raise Rejected("superseded", m.hash)
        table[key] = m

    def _reaction(self, m: Message, b: dict) -> None:
        kind, target = b.get("kind"), b.get("target")
        if kind not in REACTIONS:
            raise Rejected("unknown_reaction", str(kind))
        if target not in self.casts:
            raise Rejected("unknown_target", str(target))
        self._lww(self.reactions, (m.fid, kind, target), m)

    _reaction_add = _reaction
    _reaction_remove = _reaction

    def _link(self, m: Message, b: dict) -> None:
        kind, target = b.get("kind"), b.get("target")
        if kind not in LINKS:
            raise Rejected("unknown_link", str(kind))
        if target not in self.names or target == m.fid:
            raise Rejected("bad_link_target", str(target))
        self._lww(self.links, (m.fid, kind, target), m)

    _link_add = _link
    _link_remove = _link

    def _channel_add(self, m: Message, b: dict) -> None:
        name = str(b.get("name", ""))
        if not name or not name.replace("-", "").isalnum() or name != name.lower():
            raise Rejected("bad_channel_name", name)
        if name in self.channels:
            raise Rejected("channel_exists", name)
        if len(self.channels) >= self.limits.max_channels:
            raise Rejected("channel_quota", str(self.limits.max_channels))
        self.channels[name] = Channel(name, str(b.get("purpose", "")), m.fid, members={m.fid})

    def _channel_join(self, m: Message, b: dict) -> None:
        channel = self.channels.get(str(b.get("name")))
        if channel is None:
            raise Rejected("unknown_channel", str(b.get("name")))
        if m.fid in channel.members:
            raise Rejected("already_member", channel.name)
        channel.members.add(m.fid)

    def _channel_leave(self, m: Message, b: dict) -> None:
        channel = self.channels.get(str(b.get("name")))
        if channel is None or m.fid not in channel.members or channel.name == GENERAL:
            raise Rejected("cannot_leave", str(b.get("name")))
        channel.members.discard(m.fid)

    def _doc_set(self, m: Message, b: dict) -> None:
        channel = self.channels.get(str(b.get("name")))
        if channel is None or m.fid not in channel.members:
            raise Rejected("not_a_member", str(b.get("name")))
        text = str(b.get("text", ""))
        if len(text.encode()) > self.limits.max_doc_bytes:
            raise Rejected("doc_too_long")
        channel.doc, channel.doc_order = text, m.order

    def _answer(self, m: Message, b: dict) -> None:
        words = tuple(str(w) for w in b.get("words", ()))
        if not words:
            raise Rejected("empty_answer")
        self.answers[m.fid] = words

    # ---- reads -------------------------------------------------------------

    def live(self, table: dict) -> list[Message]:
        return [m for m in table.values() if m.type in ADDS]

    def following(self, fid: int) -> set[int]:
        return {m.data["target"] for m in self.live(self.links) if m.fid == fid}

    def edges(self) -> list[Edge]:
        """The board as a directed multigraph over fids: parallel edges of
        different kinds (and repeated edges of one kind) between one pair."""
        out: list[Edge] = []
        for m in self.live(self.links):
            out.append(Edge(m.fid, m.data["target"], m.data["kind"], m.hash))
        for cast in self.casts.values():
            if cast.parent is not None and cast.parent in self.casts:
                out.append(Edge(cast.fid, self.casts[cast.parent].fid, "reply", cast.hash))
            for fid in cast.mentions:
                out.append(Edge(cast.fid, fid, "mention", cast.hash))
        for m in self.live(self.reactions):
            target = self.casts.get(m.data["target"])
            if target is not None:
                out.append(Edge(m.fid, target.fid, m.data["kind"], m.hash))
        return out

    def inbox(self, fid: int, since: int, until: int | None = None) -> list[Notice]:
        """What the controller pushes to `fid` for casts with since <= timestamp < until:
        mentions first, then replies in threads it wrote in, then new topics in its
        channels, then casts by those it follows, then casts they recast in the window;
        newest first within a reason, capped at `limits.inbox`."""
        rank = {"mention": 0, "reply": 1, "topic": 2, "follow": 3, "recast": 4}
        follows = self.following(fid)
        picked: dict[str, Notice] = {}
        for cast in self.casts.values():
            if cast.fid == fid or cast.timestamp < since:
                continue
            if until is not None and cast.timestamp >= until:
                continue
            reasons = []
            if fid in cast.mentions:
                reasons.append("mention")
            if cast.parent is not None and fid in self.threads.get(cast.root, ()):
                reasons.append("reply")
            if cast.parent is None and fid in self.channels[cast.channel].members:
                reasons.append("topic")
            if cast.fid in follows:
                reasons.append("follow")
            if reasons:
                picked[cast.hash] = Notice(min(reasons, key=rank.get), cast, cast.timestamp)
        for m in self.live(self.reactions):
            cast = self.casts.get(m.data["target"])
            if m.data["kind"] != "recast" or m.fid not in follows or cast is None:
                continue
            if cast.fid == fid or cast.hash in picked or m.timestamp < since:
                continue
            if until is None or m.timestamp < until:
                picked[cast.hash] = Notice("recast", cast, m.timestamp)
        notices = sorted(
            picked.values(),
            key=lambda n: (rank[n.reason], -n.at, n.cast.hash),
        )
        return notices[: self.limits.inbox]

    def resolve(self, prefix: str) -> str | None:
        """The live cast whose hash starts with `prefix` (at least 6 hex chars)."""
        if len(prefix) < 6:
            return None
        hits = [h for h in self.casts if h.startswith(prefix)]
        return hits[0] if len(hits) == 1 else None


def replay(rows: list[dict], names: dict[int, str], limits: Limits = Limits()) -> Hub:
    """Rebuild a hub from its log; every row must merge exactly as recorded."""
    hub = Hub(names, limits)
    for row in rows:
        hub.submit(Message.from_json(row))
    return hub


def _freeze(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _canonical(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=list)
