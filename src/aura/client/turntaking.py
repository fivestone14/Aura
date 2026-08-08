"""Deciding when someone has finished speaking.

The hard part is not detecting silence — it is knowing what a silence *means*. People
pause mid-thought, say "ummm", trail off and pick back up. Waiting for quiet cuts them
off; waiting longer feels dead.

Three layers, cheapest first:

1. **Is there sound?** Energy-based, microseconds, runs on every frame. `EnergyVad`.
2. **Does this silence mean done?** Prosodic — a falling contour and tapering energy mean
   finished; a held or rising contour means "still going". Needs a model. `TurnDetector`
   is the seam; `SilenceDetector` is the honest placeholder.
3. **When *will* they be done?** Predicting the end a second ahead so work can start
   before it arrives. Not implemented — needs weights.

Layer 3 is the one that matters, and the arithmetic explains why. Human turn gaps run
around 200 ms, but planning a sentence takes 600–1500 ms. A pipeline that starts at
end-of-turn cannot reach 200 ms by getting faster; the component floor forbids it. It has
to start early and discard the work when the guess was wrong (docs/DESIGN.md §9).

⚠️ Even the best published detectors cut people off on roughly one turn in ten. This
module assumes it will be wrong and makes that recoverable, rather than pretending
otherwise — `TurnEvent.RESUMED` exists precisely for that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from aura.client.audio import FRAME_MS, Frame

SPEECH_THRESHOLD = 500
"""Peak int16 amplitude above which a frame counts as speech. Roughly −36 dBFS —
comfortably above room tone, comfortably below a quiet voice."""

HANGOVER_FRAMES = 3
"""Frames of quiet tolerated before speech is considered over. Stops a stop consonant
or a breath from reading as the end of a turn."""

ENDPOINT_SILENCE_MS = 700
"""Silence before a turn is called finished, when nothing smarter is available.

Deliberately long. It is the cost of having no prosodic model: a shorter value cuts
people off mid-thought, and being slow is a better failure than being rude.
"""

EAGER_SILENCE_MS = 250
"""Silence before a *speculative* end-of-turn is announced.

Work can start here and be discarded if the user resumes — the mechanism that buys
perceived speed, at the cost of some wasted computation.
"""


class TurnEvent(Enum):
    """What just happened to the conversational floor."""

    SPEECH_STARTED = "speech_started"
    EAGER_END = "eager_end"
    """Probably finished. Start work, but be ready to throw it away."""

    CONFIRMED_END = "confirmed_end"
    """Finished. Safe to commit."""

    RESUMED = "resumed"
    """They carried on after an eager end. Discard the speculative work.

    Not an error path — this is the normal cost of guessing early, and it fires often.
    """


@dataclass(frozen=True, slots=True)
class TurnSignal:
    event: TurnEvent
    silence_ms: float
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


class Vad(ABC):
    """Is this frame speech?"""

    @abstractmethod
    def is_speech(self, frame: Frame) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...


class EnergyVad(Vad):
    """Amplitude threshold with hangover.

    Crude by design. It answers layer 1 only, and layer 1 is genuinely this simple —
    reaching for a model here would spend latency on a question a comparison answers.

    The hangover matters more than the threshold: without it, the closure before a stop
    consonant reads as silence, and every "t" ends the turn.
    """

    __slots__ = ("_hangover", "_quiet_run", "_threshold")

    def __init__(
        self, threshold: int = SPEECH_THRESHOLD, hangover_frames: int = HANGOVER_FRAMES
    ) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if hangover_frames < 0:
            raise ValueError(f"hangover_frames must be non-negative, got {hangover_frames}")
        self._threshold = threshold
        self._hangover = hangover_frames
        self._quiet_run = hangover_frames + 1

    def is_speech(self, frame: Frame) -> bool:
        if frame.peak() >= self._threshold:
            self._quiet_run = 0
            return True
        self._quiet_run += 1
        return self._quiet_run <= self._hangover

    def reset(self) -> None:
        self._quiet_run = self._hangover + 1


class TurnDetector(ABC):
    """Decides when a turn is over.

    The seam a real model plugs into. An implementation reads the waveform — prosody,
    not words — and answers layer 2 and ideally layer 3.
    """

    @abstractmethod
    def observe(self, frame: Frame) -> TurnSignal | None:
        """Feed one frame. Returns a signal when something changed, else None."""

    @abstractmethod
    def reset(self) -> None: ...


class SilenceDetector(TurnDetector):
    """Ends a turn after enough quiet. The honest placeholder.

    This is layer 1 doing layer 2's job, and it is worse in a specific way: silence
    length says nothing about whether a thought was finished. Someone pausing to think
    gets cut off; someone trailing off gets waited on. A prosodic model fixes this by
    listening to the *shape* of the ending.

    It does implement eager/confirmed correctly, so the speculative machinery around it
    is exercised and testable before any weights exist.
    """

    __slots__ = ("_confirm_ms", "_eager_fired", "_eager_ms", "_silence_ms", "_speaking", "_vad")

    def __init__(
        self,
        vad: Vad | None = None,
        *,
        eager_silence_ms: float = EAGER_SILENCE_MS,
        confirm_silence_ms: float = ENDPOINT_SILENCE_MS,
    ) -> None:
        if eager_silence_ms > confirm_silence_ms:
            raise ValueError("eager threshold must not exceed the confirmation threshold")
        self._vad = vad or EnergyVad()
        self._eager_ms = eager_silence_ms
        self._confirm_ms = confirm_silence_ms
        self._silence_ms = 0.0
        self._speaking = False
        self._eager_fired = False

    @property
    def is_speaking(self) -> bool:
        """Whether the user currently holds the floor.

        Read by the negotiator, which must never speak over someone.
        """
        return self._speaking

    def observe(self, frame: Frame) -> TurnSignal | None:
        if self._vad.is_speech(frame):
            return self._on_speech()
        return self._on_silence()

    def _on_speech(self) -> TurnSignal | None:
        resumed = self._eager_fired
        self._silence_ms = 0.0
        self._eager_fired = False

        if resumed:
            # They carried on. Whatever was started speculatively is now wrong.
            self._speaking = True
            return TurnSignal(TurnEvent.RESUMED, 0.0)

        if not self._speaking:
            self._speaking = True
            return TurnSignal(TurnEvent.SPEECH_STARTED, 0.0)
        return None

    def _on_silence(self) -> TurnSignal | None:
        if not self._speaking:
            return None

        self._silence_ms += FRAME_MS

        if self._silence_ms >= self._confirm_ms:
            self._speaking = False
            self._eager_fired = False
            return TurnSignal(TurnEvent.CONFIRMED_END, self._silence_ms, confidence=0.9)

        if self._silence_ms >= self._eager_ms and not self._eager_fired:
            self._eager_fired = True
            # Low confidence on purpose: this is a guess, and the caller should treat
            # the resulting work as disposable.
            return TurnSignal(TurnEvent.EAGER_END, self._silence_ms, confidence=0.4)

        return None

    def reset(self) -> None:
        self._vad.reset()
        self._silence_ms = 0.0
        self._speaking = False
        self._eager_fired = False
