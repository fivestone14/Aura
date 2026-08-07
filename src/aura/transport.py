"""The contract between the two halves.

In production the client is on a phone and the server is on a host somewhere else, so
the client cannot import server internals — it can only send a message and wait. This
module defines what crosses that gap.

The shapes here are deliberately plain: primitives and small dataclasses, nothing that
implies shared memory or a shared Python version. If a field cannot be serialised to
JSON, it does not belong in a request or a response.

**What crosses: text and prosody numbers. Never audio** (docs/DESIGN.md §7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aura.types import ProsodyDelta, ProsodyTarget


@dataclass(frozen=True, slots=True)
class ThinkRequest:
    """Everything the server needs to produce a reply.

    Note what is absent: no audio, no acoustic feature vector, no speaker identity
    beyond an opaque key. The server is given a transcript and five numbers describing
    how it was said.
    """

    text: str
    session_key: str
    """Opaque, supplied by the client from its device certificate. The server never
    derives identity from anything else (docs/DESIGN.md §3)."""

    delta: ProsodyDelta | None = None
    """How this turn compared to the speaker's own baseline, or None while cold."""

    is_partial: bool = False
    """True when fired speculatively before the user finished. The server may still
    answer; the client decides whether to use it (docs/DESIGN.md §9)."""

    def to_wire(self) -> dict[str, Any]:
        """JSON-serialisable form. Explicit rather than reflective, so adding a field
        to the dataclass cannot silently start sending it over the network."""
        payload: dict[str, Any] = {
            "text": self.text,
            "session_key": self.session_key,
            "is_partial": self.is_partial,
        }
        if self.delta is not None:
            payload["delta"] = {
                "rate_z": self.delta.rate_z,
                "energy_z": self.delta.energy_z,
                "pitch_z": self.delta.pitch_z,
                "pitch_range_z": self.delta.pitch_range_z,
                "pause_z": self.delta.pause_z,
            }
        return payload

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> ThinkRequest:
        raw = payload.get("delta")
        return cls(
            text=payload["text"],
            session_key=payload["session_key"],
            delta=ProsodyDelta(**raw) if raw else None,
            is_partial=bool(payload.get("is_partial", False)),
        )


@dataclass(frozen=True, slots=True)
class ThinkResponse:
    """What comes back.

    `text` may be empty — the council can fail entirely, and the client must handle
    that rather than being handed something invented to fill the gap.
    """

    text: str
    prosody: ProsodyTarget = field(default_factory=ProsodyTarget)
    rationale: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def to_wire(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "rationale": self.rationale,
            "prosody": {
                "rate_scale": self.prosody.rate_scale,
                "pitch_shift_semitones": self.prosody.pitch_shift_semitones,
                "energy_scale": self.prosody.energy_scale,
                "pause_scale": self.prosody.pause_scale,
                "register": self.prosody.register.value,
            },
        }

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> ThinkResponse:
        from aura.types import Register

        raw = payload.get("prosody") or {}
        return cls(
            text=payload["text"],
            rationale=payload.get("rationale", ""),
            prosody=ProsodyTarget(
                rate_scale=raw.get("rate_scale", 1.0),
                pitch_shift_semitones=raw.get("pitch_shift_semitones", 0.0),
                energy_scale=raw.get("energy_scale", 1.0),
                pause_scale=raw.get("pause_scale", 1.0),
                register=Register(raw.get("register", "neutral")),
            ),
        )


class Transport(ABC):
    """How the client reaches the server.

    One method, because the client's needs are genuinely that narrow. Anything richer
    would leak server structure across a boundary that is a network hop in production.
    """

    @abstractmethod
    async def think(self, request: ThinkRequest) -> ThinkResponse:
        """Send a turn and wait for a reply.

        Implementations must not raise on a server-side failure — return an empty
        response instead. A dropped connection is a normal event in this system, and
        the client already knows how to carry on without an answer.
        """


class UnavailableTransport(Transport):
    """A server that is never reachable.

    Not a test double — this is the honest configuration for a client running with no
    host configured, and exercises the degraded path the design promises: the client
    still hears, still takes turns, still acknowledges (docs/DESIGN.md §7).
    """

    async def think(self, request: ThinkRequest) -> ThinkResponse:
        del request
        return ThinkResponse(text="", rationale="no server configured")
