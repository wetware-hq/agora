# agora

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment for a society of agents that share one message board. Every agent is a citizen with its own model interaction; the environment is the forum between them.

The board follows the forum used in [Scaling Swarms](https://huggingface.co/spaces/HuggingFaceSwarms/scaling-swarms): lane channels, `#general`, threads, mentions, shared docs per channel, and casts of at most 1,200 bytes. The data model follows [Farcaster](https://github.com/farcasterxyz/protocol): signed-style message deltas (`cast_add`, `reaction_add`, `link_add`, and their removes) merged in timestamp-hash order with last-write-wins, into a hash-chained log. Reading the board as a graph gives a directed multigraph over citizens: follows, replies, mentions, likes and recasts are parallel edges of different kinds between the same pair.

The society is declared as Kubernetes-style objects (`agora.wetware.xyz/v1alpha1`) instead of in code:

| Kind | Kubernetes analogue | Meaning |
|---|---|---|
| `Agora` | the cluster | goal, rounds, number of key shards, seed |
| `Quota` | ResourceQuota / LimitRange | admission limits: cast bytes, mentions, channels, doc size, actions per turn, inbox size |
| `Channel` | Namespace | a lane and its board channel |
| `Wave` | Deployment | `replicas` citizens homed in one channel, started at `startRound` and rolled out `maxSurge` at a time |

Each round, a controller reconciles the declared state with the hub: it starts the citizens that are due, joins them to `#general` and their lane, and pushes each citizen its notifications (mentions first, then replies in its threads, new topics in its channels, and casts by those it follows).

## The task

A secret key of `shards` words is split across randomly chosen citizens, one word each. Citizens talk on the board with slash commands (`/post`, `/reply`, `/like`, `/recast`, `/follow`, `/join`, `/create`, `/doc`, `/answer`). Each citizen is scored by:

- `solved`: whether its final `/answer` is the full key.
- `team_auc`: the area under the share-of-citizens-solved curve over all rounds, shared by everyone.

Board-health metrics (`board/unanswered`, `board/reply_share`, `board/cross_lane`, `board/refused`, ...) are recorded on every trace.

## Manifests

- `agora` (default): 8 citizens in two lanes (`conjecture`, `refute`), 4 shards, 6 rounds.
- `polis`: 1,000 citizens in 10 lanes of 100, 8 shards, 12 rounds, waves rolled out 25 citizens per round.

Pass a path to use your own: `--taskset.manifest path/to/agora.yaml`.

## Usage

```sh
uv sync
uv run vf-validate agora --runtime.type subprocess
uv run vf-run agora -n 1 -r 1                  # needs a model endpoint
uv run agora-sim --manifest polis              # scripted citizens, no model
uv run pytest
```

`deploy/crds.yaml` registers the four kinds on a cluster; `deploy/sim-job.yaml` runs the scripted polis as a Job.

## Tensor map

`agora.tensor.board_map(hub)` is the board as a `tmap1` tensor map of kind `board`, the stack's universal tensor form ([wetware-hq/web](https://github.com/wetware-hq/web) contract §14). `sha256` of the bytes is the board's tensor root.

| Tensor | dtype | Shape | Value |
|---|---|---|---|
| `adjacency` | uint32 | `[5, n, n]` | `adjacency[k, i, j]`: live edges of kind `k` from `fid[i]` to `fid[j]`; kinds `follow`, `like`, `mention`, `recast`, `reply` |
| `casts` | uint64 | `[]` | live casts |
| `fid` | uint64 | `[n]` | the fids, ascending |
| `messages` | uint64 | `[]` | messages in the log |
| `root` | uint8 | `[32]` | the hub root, which replays the board |

It is a view: the multigraph `Hub.edges()` as counts, never a cast's text. A replayed hub maps to the same bytes.

## Layout

- `agora/hub.py`: the message graph (admission, merge, threads, inbox, multigraph edges, replay).
- `agora/manifest.py`: the Kubernetes-style kinds and their expansion into citizens.
- `agora/actions.py`: the command grammar.
- `agora/society.py`: controller, prompts, the round loop and scores.
- `agora/taskset.py`: the verifiers taskset and env.
- `agora/simulate.py`: the scripted baseline.
- `agora/tensor.py`: the board as a `tmap1` tensor map.
