"""The server side of the wire: request in, reply out.

Everything the server does for one turn, behind the `Transport` interface so the client
cannot tell whether it is talking to an in-process object or a machine across the
network.

Also the place where per-session state lives. Baselines are the client's business — the
client owns the audio — but profiles belong here, because they persist across sessions
and devices.
"""

from __future__ import annotations

from aura.server.chair import Chair
from aura.server.council import Council
from aura.server.profile import Profile
from aura.transport import ThinkRequest, ThinkResponse, Transport
from aura.types import ProsodyFrame, ProsodyTarget, Turn


class LocalService(Transport):
    """Runs the council and chair in-process.

    Used when client and server are the same program — during development, in tests,
    and on a single-machine deployment. A networked implementation wraps the same
    `handle` method behind a socket.
    """

    def __init__(
        self,
        council: Council,
        chair: Chair | None = None,
        *,
        profiles: dict[str, Profile] | None = None,
    ) -> None:
        self._council = council
        self._chair = chair or Chair()
        self._profiles: dict[str, Profile] = profiles if profiles is not None else {}

    def profile_for(self, session_key: str) -> Profile:
        """Get or create the profile for a session key.

        Created lazily: a first-time caller gets a neutral profile rather than an error,
        which is the cold-start behaviour the design calls for — start neutral rather
        than guess (docs/DESIGN.md §4).
        """
        if session_key not in self._profiles:
            self._profiles[session_key] = Profile(key=session_key)
        return self._profiles[session_key]

    async def think(self, request: ThinkRequest) -> ThinkResponse:
        """Handle one turn.

        Never raises. A failure here would strand a client that is mid-conversation and
        cannot do anything useful with an exception, so problems come back as an empty
        response and the client falls back to its degraded path.
        """
        try:
            return await self._handle(request)
        except Exception as exc:
            return ThinkResponse(text="", rationale=f"server error: {type(exc).__name__}: {exc}")

    async def _handle(self, request: ThinkRequest) -> ThinkResponse:
        profile = self.profile_for(request.session_key)

        turn = _turn_from(request)
        result = await self._council.deliberate(turn, profile)

        # The tone plan is computed by the client, which owns the baseline. The server
        # passes a neutral target through; the client overrides it with its own
        # decision. Sending one at all keeps the response shape stable.
        reply = self._chair.merge(result, ProsodyTarget())
        if reply is None:
            missing = ", ".join(r.value for r in result.timed_out) or "unknown"
            return ThinkResponse(text="", rationale=f"no usable reading ({missing})")

        return ThinkResponse(text=reply.text, prosody=reply.prosody, rationale=reply.rationale)


def _turn_from(request: ThinkRequest) -> Turn:
    """Rebuild a Turn from the wire request.

    The server never receives acoustics, only the delta, so the frame is a placeholder
    carrying no measurements. This is deliberate: if the server ever needs real
    acoustics, that should be a visible protocol change rather than something that
    quietly starts working.
    """
    placeholder = ProsodyFrame(
        speaking_rate=0.0, energy_db=0.0, pitch_hz=0.0, pitch_range_semitones=0.0, pause_ratio=0.0
    )
    return Turn(
        text=request.text,
        prosody=placeholder,
        delta=request.delta,
        is_partial=request.is_partial,
    )
