"""The conversation loop.

Ties the client-side components together and calls the server through `Transport`. This
is the only place that knows the order things happen in, which is the point: every other
module can be reasoned about on its own.

The order matters and is not arbitrary:

1. Remove echo, or we measure Aura instead of the user.
2. Score against this speaker's own baseline, before folding the turn in.
3. Choose the tone locally — the client owns the baseline, so it owns this decision.
4. Ask the server for words, covering the gap out loud if it takes long enough.
5. Fall back to something honest if the server has nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from aura.client.audio import AudioLoop
from aura.client.baseline import SpeakerBaseline
from aura.client.negotiator import Acknowledgement, Negotiator
from aura.client.policy import decide
from aura.transport import ThinkRequest, Transport, UnavailableTransport
from aura.types import ProsodyFrame, ProsodyTarget

FALLBACK_TEXT = "I didn't catch that — could you say it again?"
"""Used when the server returns nothing.

Deliberately an admission rather than a deflection. The alternatives are silence, which
reads as broken, or a plausible-sounding non-answer, which is worse than either.
"""


@dataclass(frozen=True, slots=True)
class SpokenReply:
    """What the client should say, and how."""

    text: str
    prosody: ProsodyTarget
    acknowledgement: Acknowledgement | None = None
    """Played immediately, before `text`, when the wait warranted covering."""

    is_fallback: bool = False
    rationale: str = ""


@dataclass
class SessionStats:
    turns: int = 0
    fallbacks: int = 0
    acknowledgements: int = 0
    total_wait_seconds: float = 0.0

    @property
    def mean_wait_seconds(self) -> float:
        return self.total_wait_seconds / self.turns if self.turns else 0.0


@dataclass
class Session:
    """One conversation with one person on one device.

    Holds the client-side state that must not be shared: the audio loop and the speaker
    baseline. The profile lives on the server, because it outlives the session.
    """

    session_key: str
    transport: Transport = field(default_factory=UnavailableTransport)
    baseline: SpeakerBaseline = field(default_factory=SpeakerBaseline)
    negotiator: Negotiator = field(default_factory=Negotiator)
    audio: AudioLoop = field(default_factory=AudioLoop)
    stats: SessionStats = field(default_factory=SessionStats)

    async def handle_turn(
        self,
        text: str,
        prosody: ProsodyFrame,
        *,
        wants_backchannel: bool = True,
    ) -> SpokenReply:
        """Process one complete thing the user said.

        `prosody` is measured from audio that has already been through the echo loop.
        Passing raw microphone acoustics here would poison the baseline with Aura's own
        voice, which is the failure the whole audio module exists to prevent.
        """
        self.stats.turns += 1

        # Score first, fold in second. Reversing this lets an unusual turn partially
        # normalise itself away.
        delta = self.baseline.observe_and_compare(prosody)

        # Tone is decided here, not on the server: the baseline is client-side, and
        # sending it upstream would mean sending a model of the user's voice.
        decision = decide(delta, profile=None)

        started = time.monotonic()
        response = await self.transport.think(
            ThinkRequest(text=text, session_key=self.session_key, delta=delta)
        )
        waited = time.monotonic() - started
        self.stats.total_wait_seconds += waited

        self.negotiator.observe_turn(seconds_since_last_ack=waited)
        ack = self._maybe_acknowledge(waited, wants_backchannel)

        if response.is_empty:
            self.stats.fallbacks += 1
            return SpokenReply(
                text=FALLBACK_TEXT,
                prosody=decision.target,
                acknowledgement=ack,
                is_fallback=True,
                rationale=response.rationale or "server returned nothing",
            )

        return SpokenReply(
            text=response.text,
            # The client's tone decision wins. The server has no baseline and therefore
            # no basis for one.
            prosody=decision.target,
            acknowledgement=ack,
            rationale=f"{decision.rationale} | {response.rationale}",
        )

    def _maybe_acknowledge(self, waited: float, wants_backchannel: bool) -> Acknowledgement | None:
        """Decide retrospectively whether the gap needed covering.

        A real client asks *before* the wait, using a predicted duration, and plays the
        token while the server works. Deciding after the fact keeps this method
        synchronous and testable; the negotiator's logic is identical either way.
        """
        decision = self.negotiator.consider(
            expected_wait_seconds=waited,
            user_still_speaking=False,
            profile_allows=wants_backchannel,
        )
        if decision.should_speak:
            self.stats.acknowledgements += 1
        return decision.acknowledgement

    @property
    def is_ready_for_live_audio(self) -> bool:
        """Whether it is safe to run with real playback.

        Without echo cancellation the microphone hears Aura, and every downstream
        component degrades silently. Worth asserting before opening a device rather
        than discovering it from confusing transcripts.
        """
        return self.audio.is_echo_protected

    @property
    def supports_two_channel_turn_taking(self) -> bool:
        """Whether a turn-taking model can get both channels here.

        Under WebRTC, echo cancellation happens upstream and consumes the reference
        signal, so the application only ever sees the cleaned microphone channel. A
        model that wants to hear both sides has to source Aura's own audio from the
        synthesis path instead.

        This is a wiring-time question, so it is answerable at wiring time rather than
        manifesting later as a model that silently receives one channel of silence.
        """
        return self.audio.has_reference_signal

    def reset(self) -> None:
        """Clear everything session-scoped. The server-side profile is untouched."""
        self.baseline = SpeakerBaseline()
        self.negotiator.reset()
        self.audio.reset()
        self.stats = SessionStats()
