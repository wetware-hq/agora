import struct

import pytest

from agora.hub import Hub, Limits, Message, Rejected, replay
from agora.manifest import load
from agora.simulate import gossip
from agora.society import Society
from agora.taskset import AgoraTaskset, manifest_docs
from agora.tensor import KINDS, MAX_BYTES, board_map, tensor_root

NAMES = {1: "ada", 2: "bo", 3: "cy"}


def run(name: str, seed: int = 0) -> Society:
    society = Society(load(manifest_docs(name)), seed)
    memory: dict = {}
    for r in range(society.spec.agora.rounds):
        society.reconcile(r)
        society.step(
            r,
            {
                a.fid: gossip(society, a.fid, society.prompt(a, r), memory.setdefault(a.fid, {}))
                for a in society.running()
            },
        )
    return society


def test_threads_mentions_and_multigraph():
    hub = Hub(NAMES)
    for t, fid in enumerate(NAMES, 1):
        hub.submit(Message.make("channel_join", fid, t, name="general"))
    root = Message.make("cast_add", 1, 10, text="hello @bo", channel="general", mentions=[2])
    hub.submit(root)
    hub.submit(Message.make("cast_add", 2, 11, text="hi", parent=root.hash, mentions=[]))
    hub.submit(Message.make("link_add", 3, 12, kind="follow", target=1))
    hub.submit(Message.make("reaction_add", 3, 13, kind="like", target=root.hash))
    kinds = sorted((e.src, e.dst, e.kind) for e in hub.edges())
    assert kinds == [
        (1, 2, "mention"),
        (2, 1, "reply"),
        (3, 1, "follow"),
        (3, 1, "like"),
    ]
    assert next(n.reason for n in hub.inbox(2, 0)) == "mention"
    assert replay([m.to_json() for m in hub.log], NAMES).root == hub.root


def test_admission_limits():
    hub = Hub(NAMES, Limits(max_cast_bytes=5))
    hub.submit(Message.make("channel_join", 1, 1, name="general"))
    with pytest.raises(Rejected, match="too_long"):
        hub.submit(Message.make("cast_add", 1, 2, text="too long", channel="general", mentions=[]))
    assert (
        hub.offer(Message.make("cast_add", 2, 3, text="hi", channel="general", mentions=[]))
        == "not_a_member"
    )
    assert (
        hub.offer(Message.make("cast_add", 1, 0, text="hi", channel="general", mentions=[]))
        == "out_of_order"
    )


def test_link_remove_wins():
    hub = Hub(NAMES)
    hub.submit(Message.make("link_add", 1, 1, kind="follow", target=2))
    hub.submit(Message.make("link_remove", 1, 2, kind="follow", target=2))
    assert hub.edges() == [] and hub.following(1) == set()


def test_staggered_waves():
    spec = load(manifest_docs("polis"))
    starts = [a.start for a in spec.agents()]
    assert len(starts) == 1000 and starts[:100] == [0] * 25 + [1] * 25 + [2] * 25 + [3] * 25
    assert max(starts) == 5


@pytest.mark.parametrize("name", ["agora", "polis"])
def test_scripted_society_solves(name):
    society = run(name)
    assert society.utility[-1] == 1.0 and 0 < society.auc() < 1
    assert society.health()["refused"] == 0
    assert run(name).hub.root == society.hub.root


def test_taskset_loads():
    tasks = (
        AgoraTaskset(AgoraTaskset.config_type()(num_tasks=3)).load()
        if hasattr(AgoraTaskset, "config_type")
        else None
    )
    if tasks is not None:
        assert len(tasks) == 3


def test_shards_only_go_to_citizens_who_start():
    docs = manifest_docs("agora")
    for d in docs:
        if d["kind"] == "Wave" and d["metadata"]["name"] == "critic":
            d["spec"]["startRound"] = 99
    society = Society(load(docs), 0)
    starts = {a.fid: a.start for a in society.agents}
    assert all(starts[fid] < society.spec.agora.rounds for fid in society.shard)


def test_recast_reaches_followers():
    hub = Hub(NAMES)
    for t, fid in enumerate(NAMES, 1):
        hub.submit(Message.make("channel_join", fid, t, name="general"))
    hub.submit(Message.make("channel_add", 1, 4, name="lane", purpose=""))
    cast = Message.make("cast_add", 1, 5, text="news", channel="lane", mentions=[])
    hub.submit(cast)
    hub.submit(Message.make("link_add", 3, 6, kind="follow", target=2))
    hub.submit(Message.make("reaction_add", 2, 7, kind="recast", target=cast.hash))
    assert [(n.reason, n.cast.hash) for n in hub.inbox(3, 6)] == [("recast", cast.hash)]


def test_dotted_names_are_refused():
    docs = manifest_docs("agora")
    for d in docs:
        if d["kind"] == "Wave" and d["metadata"]["name"] == "critic":
            d["metadata"]["name"] = "crit.ic"
    with pytest.raises(ValueError, match="must match"):
        load(docs)


def test_board_tensor_map():
    hub = Hub(NAMES)
    for t, fid in enumerate(NAMES, 1):
        hub.submit(Message.make("channel_join", fid, t, name="general"))
    root = Message.make("cast_add", 1, 10, text="hello @bo", channel="general", mentions=[2])
    hub.submit(root)
    hub.submit(Message.make("cast_add", 2, 11, text="hi", parent=root.hash, mentions=[]))
    hub.submit(Message.make("link_add", 3, 12, kind="follow", target=1))
    hub.submit(Message.make("reaction_add", 3, 13, kind="like", target=root.hash))
    blob = board_map(hub)
    # bytes web ref/tensor.py unpack reads and pack rewrites identically
    assert (len(blob), tensor_root(hub)) == (
        329,
        "b3adaa37adbd4be54562ba19438fc7d29af932b5a2fb479e21003c760ca1ff0e",
    )
    assert blob[:13] == b"tmap1\x05board\x05\x00"
    adjacency = struct.unpack("<45I", blob[37 : 37 + 180])
    cell = {
        (k, i, j): adjacency[(k * 3 + i) * 3 + j]
        for k in range(5)
        for i in range(3)
        for j in range(3)
    }
    on = {(KINDS[k], i + 1, j + 1) for (k, i, j), v in cell.items() if v}
    assert on == {(e.kind, e.src, e.dst) for e in hub.edges()}
    assert b"hello" not in blob
    assert replay([m.to_json() for m in hub.log], NAMES).root == hub.root
    assert board_map(replay([m.to_json() for m in hub.log], NAMES)) == blob
    hub.submit(Message.make("link_remove", 3, 14, kind="follow", target=1))
    assert tensor_root(hub) != "b3adaa37adbd4be54562ba19438fc7d29af932b5a2fb479e21003c760ca1ff0e"


def test_board_over_the_map_limit_is_refused():
    n = 7400
    assert 4 * len(KINDS) * n * n > MAX_BYTES
    with pytest.raises(ValueError, match="too_large"):
        tensor_root(Hub({fid: f"u{fid}" for fid in range(1, n + 1)}))
