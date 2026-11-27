"""Kubernetes-style objects for an agora, and the controller that reconciles them.

An agora is declared, not scripted: `Agora` is the cluster (goal, rounds, shards),
`Quota` is its ResourceQuota/LimitRange (the hub's admission limits), `Channel` is a
namespace (a lane and its board channel), and `Wave` is a Deployment whose pods are
agents, rolled out in staggered batches. The controller compares this desired state
with the hub each round and acts: it starts due agents, places them in their
channels, and pushes their notifications.
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agora.hub import GENERAL, Limits

API_VERSION = "agora.wetware.xyz/v1alpha1"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Metadata(Strict):
    name: str
    labels: dict[str, str] = {}


class AgoraSpec(Strict):
    goal: str
    rounds: int = Field(6, ge=1)
    shards: int = Field(4, ge=1, le=16)
    seed: int = 0


class QuotaSpec(Strict):
    max_cast_bytes: int = Field(1200, alias="maxCastBytes", ge=1)
    max_mentions: int = Field(10, alias="maxMentions", ge=0)
    max_channels: int = Field(40, alias="maxChannels", ge=1)
    max_doc_bytes: int = Field(2000, alias="maxDocBytes", ge=0)
    max_actions: int = Field(8, alias="maxActionsPerTurn", ge=1)
    inbox: int = Field(20, ge=1)


class ChannelSpec(Strict):
    purpose: str = ""


class RolloutSpec(Strict):
    max_surge: int = Field(0, alias="maxSurge", ge=0)
    """Agents started per batch; 0 starts the whole wave at once."""
    interval: int = Field(1, ge=1)
    """Rounds between batches."""


class WaveSpec(Strict):
    replicas: int = Field(ge=1)
    channel: str
    start_round: int = Field(0, alias="startRound", ge=0)
    rollout: RolloutSpec = RolloutSpec()
    role: str = ""


class Object(Strict):
    api_version: Literal["agora.wetware.xyz/v1alpha1"] = Field(alias="apiVersion")
    kind: Literal["Agora", "Quota", "Channel", "Wave"]
    metadata: Metadata
    spec: dict = {}


class Agent(Strict):
    fid: int
    name: str
    wave: str
    channel: str
    start: int
    role: str = ""


class Spec(Strict):
    name: str
    agora: AgoraSpec
    quota: QuotaSpec = QuotaSpec()
    channels: dict[str, ChannelSpec] = {}
    waves: dict[str, WaveSpec] = {}

    @model_validator(mode="after")
    def _check(self) -> Spec:
        if not self.waves:
            raise ValueError("an agora needs at least one Wave")
        for name, wave in self.waves.items():
            if wave.channel != GENERAL and wave.channel not in self.channels:
                raise ValueError(f"Wave {name} names unknown Channel {wave.channel}")
        if len(self.channels) + 1 > self.quota.max_channels:
            raise ValueError("Channels exceed the Quota's maxChannels")
        if sum(a.start < self.agora.rounds for a in self.agents()) < self.agora.shards:
            raise ValueError("fewer agents start before the last round than there are shards")
        return self

    @property
    def limits(self) -> Limits:
        q = self.quota
        return Limits(q.max_cast_bytes, q.max_mentions, q.max_channels, q.max_doc_bytes, q.inbox)

    def agents(self) -> list[Agent]:
        """Every pod of every Wave, in declaration order: fid k is the k-th agent."""
        out: list[Agent] = []
        for name, wave in self.waves.items():
            batch = wave.rollout.max_surge or wave.replicas
            for i in range(wave.replicas):
                start = wave.start_round + (i // batch) * wave.rollout.interval
                out.append(
                    Agent(
                        fid=len(out) + 1,
                        name=f"{name}-{i}",
                        wave=name,
                        channel=wave.channel,
                        start=start,
                        role=wave.role,
                    )
                )
        return out


def load(source: str | list[dict]) -> Spec:
    """Parse a multi-document manifest (YAML text or already-loaded objects)."""
    docs = [d for d in (yaml.safe_load_all(source) if isinstance(source, str) else source) if d]
    objects = [Object.model_validate(d) for d in docs]
    agoras = [o for o in objects if o.kind == "Agora"]
    if len(agoras) != 1:
        raise ValueError("a manifest declares exactly one Agora")
    quotas = [o for o in objects if o.kind == "Quota"]
    if len(quotas) > 1:
        raise ValueError("a manifest declares at most one Quota")
    return Spec(
        name=agoras[0].metadata.name,
        agora=AgoraSpec.model_validate(agoras[0].spec),
        quota=QuotaSpec.model_validate(quotas[0].spec if quotas else {}),
        channels={
            o.metadata.name: ChannelSpec.model_validate(o.spec)
            for o in objects
            if o.kind == "Channel"
        },
        waves={
            o.metadata.name: WaveSpec.model_validate(o.spec) for o in objects if o.kind == "Wave"
        },
    )
