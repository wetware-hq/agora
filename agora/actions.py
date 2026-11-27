"""The agent's side of the board: one slash command per line."""

from __future__ import annotations

import re
from dataclasses import dataclass

GRAMMAR = """\
/post #channel <text>      start a topic in a channel you are in
/reply <hash> <text>       reply in a thread (use the 6+ char hash shown in [brackets])
/like <hash>               react to a cast; /recast <hash> boosts it to your followers
/follow @name              get that agent's casts in your notifications; /unfollow @name
/join #channel             join a channel; /leave #channel
/create #channel <purpose> create a channel (the quota is small: reuse before you create)
/doc #channel <text>       replace a channel's shared doc (keep it short and current)
/answer <word> <word> ...  your current answer: every shard word, in shard order
Write @name inside a cast to mention someone; only they are woken by it."""

COMMAND = re.compile(r"^\s*/([a-z]+)\b\s*(.*)$")
CHANNEL = re.compile(r"^#([a-z0-9-]+)\s*(.*)$", re.DOTALL)
MENTION = re.compile(r"@([a-z0-9][a-z0-9-]*)")


@dataclass(frozen=True)
class Action:
    verb: str
    args: tuple[str, ...]


def parse(reply: str, limit: int) -> tuple[list[Action], int]:
    """The reply's commands (at most `limit`) and the number of unparseable ones."""
    actions: list[Action] = []
    bad = 0
    for line in reply.splitlines():
        match = COMMAND.match(line)
        if match is None:
            continue
        verb, rest = match.group(1), match.group(2).strip()
        action = _action(verb, rest)
        if action is None:
            bad += 1
        elif len(actions) < limit:
            actions.append(action)
        else:
            bad += 1
    return actions, bad


def _action(verb: str, rest: str) -> Action | None:
    if verb in {"post", "create", "doc"}:
        m = CHANNEL.match(rest)
        if m is None or (verb == "post" and not m.group(2).strip()):
            return None
        return Action(verb, (m.group(1), m.group(2).strip()))
    if verb in {"join", "leave"}:
        m = CHANNEL.match(rest)
        return Action(verb, (m.group(1),)) if m and not m.group(2).strip() else None
    if verb == "reply":
        head, _, text = rest.partition(" ")
        head = head.strip("[]")
        return Action(verb, (head, text.strip())) if head and text.strip() else None
    if verb in {"like", "recast"}:
        return Action(verb, (rest.strip("[] "),)) if rest else None
    if verb in {"follow", "unfollow"}:
        name = rest.lstrip("@").strip()
        return Action(verb, (name,)) if name and " " not in name else None
    if verb == "answer":
        words = tuple(w.strip(",.;").lower() for w in rest.split())
        return Action(verb, words) if words else None
    return None


def mentions(text: str) -> list[str]:
    return list(dict.fromkeys(MENTION.findall(text.lower())))
