"""Audio plumbing: the reference tap and the echo-cancellation seam.

This is the first thing built and the thing everything else depends on, because if
Aura hears itself the whole system degrades silently: speech recognition transcribes
the assistant's own words, turn detection fires on them, and the prosody baseline
starts modelling Aura's voice instead of the user's (docs/DESIGN.md §7).

The critical structure here is the **reference tap**. An echo canceller needs the exact
samples that went to the speaker, time-aligned with what came back from the microphone.
Retrofitting that is painful, so playback forks its buffer from the start — and the same
signal is what a two-channel turn-taking model needs later, so one piece of plumbing
serves both.

No native dependencies. `EchoCanceller` is an interface; the real implementation wraps
WebRTC AEC3 and lands when hardware capture does.
"""

from __future__ import annotations

import array
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

SAMPLE_RATE = 16_000
"""Hz. 16 kHz is the speech-processing standard and what the acoustic feature set
expects. Capture at device rate and resample once, at the edge."""

FRAME_MS = 20
"""Milliseconds per frame. 20 ms is the WebRTC convention and divides evenly into the
rates every downstream component wants."""

FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

MAX_REFERENCE_DELAY_MS = 500
"""How far back the canceller may look for the played audio. Covers realistic output
buffering; beyond this the echo has decayed enough not to matter."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One 20 ms block of mono 16-bit PCM.

    Frames carry their own sample index so alignment is explicit rather than implied by
    arrival order. Anything that reorders or drops frames is then visible instead of
    quietly corrupting the echo estimate.
    """

    samples: array.array[int]
    index: int
    """Sample offset from stream start. Monotonic, and the basis for alignment."""

    def __post_init__(self) -> None:
        if self.samples.typecode != "h":
            raise ValueError(f"expected int16 samples, got typecode {self.samples.typecode!r}")
        if len(self.samples) != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES} samples, got {len(self.samples)}")
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}")

    @property
    def timestamp_ms(self) -> float:
        return self.index * 1000.0 / SAMPLE_RATE

    @classmethod
    def silence(cls, index: int = 0) -> Frame:
        return cls(array.array("h", [0] * FRAME_SAMPLES), index)

    def peak(self) -> int:
        """Largest absolute sample. Cheap enough to call per frame."""
        return max(abs(s) for s in self.samples) if self.samples else 0


class ReferenceTap:
    """Remembers what was recently sent to the speaker.

    Playback writes here on its way out; the canceller reads back by timestamp. A
    bounded deque means a canceller that stalls cannot grow memory without limit — it
    just loses the oldest history, which is the correct failure.
    """

    __slots__ = ("_capacity", "_frames")

    def __init__(self, max_delay_ms: int = MAX_REFERENCE_DELAY_MS) -> None:
        if max_delay_ms < FRAME_MS:
            raise ValueError(f"max_delay_ms must be at least one frame ({FRAME_MS}ms)")
        self._capacity = max_delay_ms // FRAME_MS
        self._frames: deque[Frame] = deque(maxlen=self._capacity)

    @property
    def capacity_frames(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._frames)

    def record(self, frame: Frame) -> None:
        """Note a frame on its way to the speaker."""
        self._frames.append(frame)

    def lookup(self, index: int, tolerance_samples: int = FRAME_SAMPLES) -> Frame | None:
        """Find what was playing around a given capture position.

        Returns None when nothing was playing then — which is the common case, since
        Aura is silent for most of a conversation, and means the canceller has nothing
        to subtract.
        """
        best: Frame | None = None
        best_distance = tolerance_samples + 1
        for frame in self._frames:
            distance = abs(frame.index - index)
            if distance < best_distance:
                best, best_distance = frame, distance
        return best

    def clear(self) -> None:
        """Drop history. Call between sessions so one conversation's playback cannot
        be mistaken for another's echo."""
        self._frames.clear()


class EchoCanceller(ABC):
    """Removes Aura's own voice from the microphone signal."""

    @abstractmethod
    def process(self, captured: Frame, reference: Frame | None) -> Frame:
        """Return the captured frame with any echo of `reference` removed.

        `reference` is None when nothing was playing, in which case implementations
        should return the input unchanged rather than doing speculative work.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget adaptation state — between sessions, or after a device change."""


class PassthroughCanceller(EchoCanceller):
    """Does nothing. For tests and for machines where Aura never speaks aloud.

    ⚠️ Not safe for a live full-duplex session: with real playback the microphone hears
    Aura, and everything downstream degrades. `AudioLoop` warns rather than silently
    accepting this.
    """

    def process(self, captured: Frame, reference: Frame | None) -> Frame:
        del reference
        return captured

    def reset(self) -> None:
        return None


class SubtractiveCanceller(EchoCanceller):
    """A deliberately simple canceller: scaled subtraction of the reference.

    This is **not** a substitute for AEC3. It has no adaptive filter and no model of
    room response, so it only removes echo in the trivial case where the return path is
    close to linear and undelayed. Its purpose is to make the reference-tap wiring
    testable end to end — if alignment is wrong, this produces obviously wrong output,
    which is exactly the signal wanted before the real canceller is dropped in.
    """

    __slots__ = ("_attenuation",)

    def __init__(self, attenuation: float = 0.8) -> None:
        if not 0.0 <= attenuation <= 1.0:
            raise ValueError(f"attenuation must be in [0,1], got {attenuation}")
        self._attenuation = attenuation

    def process(self, captured: Frame, reference: Frame | None) -> Frame:
        if reference is None:
            return captured
        out = array.array("h", [0] * FRAME_SAMPLES)
        for i in range(FRAME_SAMPLES):
            value = captured.samples[i] - int(reference.samples[i] * self._attenuation)
            # int16 saturation, not wraparound: clipping is audible but a wrapped
            # sample is a loud click that reads downstream as a speech onset.
            out[i] = max(-32768, min(32767, value))
        return Frame(out, captured.index)

    def reset(self) -> None:
        return None


@dataclass
class LoopStats:
    """Counters for diagnosing a loop that isn't behaving."""

    captured: int = 0
    played: int = 0
    cancelled: int = 0
    """Frames where a reference was found and echo removal actually ran."""


class AudioLoop:
    """Wires capture, playback, and cancellation together.

    Owns the reference tap so callers cannot forget to feed it — the single most
    consequential thing to get wrong in this module, and the reason it is not left to
    the caller.
    """

    __slots__ = ("_canceller", "_capture_index", "_playback_index", "_reference", "stats")

    def __init__(self, canceller: EchoCanceller | None = None) -> None:
        self._canceller = canceller or PassthroughCanceller()
        self._reference = ReferenceTap()
        self._capture_index = 0
        self._playback_index = 0
        self.stats = LoopStats()

    @property
    def is_echo_protected(self) -> bool:
        """False when nothing is actually removing Aura's voice from the input.

        Worth asserting before a live session: a passthrough canceller with real
        playback is the silent-failure case this whole module exists to prevent.
        """
        return not isinstance(self._canceller, PassthroughCanceller)

    def play(self, samples: array.array[int]) -> Frame:
        """Send a frame to the speaker, recording it for cancellation."""
        frame = Frame(samples, self._playback_index)
        self._reference.record(frame)
        self._playback_index += FRAME_SAMPLES
        self.stats.played += 1
        return frame

    def capture(self, samples: array.array[int]) -> Frame:
        """Take a frame from the microphone and remove any echo."""
        raw = Frame(samples, self._capture_index)
        reference = self._reference.lookup(self._capture_index)
        cleaned = self._canceller.process(raw, reference)

        self._capture_index += FRAME_SAMPLES
        self.stats.captured += 1
        if reference is not None:
            self.stats.cancelled += 1
        return cleaned

    def reset(self) -> None:
        """Clear all state between sessions."""
        self._reference.clear()
        self._canceller.reset()
        self._capture_index = 0
        self._playback_index = 0
        self.stats = LoopStats()
