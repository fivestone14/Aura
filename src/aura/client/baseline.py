"""Per-speaker acoustic baselines.

The first principle in practice: we never ask "what kind of voice is this?", only
"how does this turn compare to the last N turns from the same person?".

The window is bounded so the baseline tracks a speaker who is genuinely changing —
tired, in a different room, a different time of day — rather than averaging over
their entire history forever. Recomputing over a small deque is cheap enough at this
window size that an incremental estimator would be premature optimisation, and it
avoids the numerical drift those accumulate.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from aura.types import ProsodyDelta, ProsodyFrame

DEFAULT_WINDOW = 50
"""Turns retained. Long enough to be stable, short enough to track real change."""

MIN_SAMPLES = 5
"""Below this, a standard deviation is noise. We return no delta rather than a bad one."""

# Minimum plausible spread per dimension, in that dimension's natural units.
#
# Variance flooring is standard practice in speech processing, and it fixes a real
# failure mode here: a speaker who happens to be very consistent produces a near-zero
# standard deviation, and dividing by it yields either infinity or — if naively
# guarded — zero. Zero is the dangerous one, because it silently reports "completely
# normal" for a turn that is wildly outside their range.
#
# Values are conservative estimates of how much a person varies turn to turn even when
# nothing is going on. They act only when measured spread falls below them.
_VARIANCE_FLOOR: dict[str, float] = {
    "rate": 0.25,  # syllables/sec
    "energy": 1.5,  # dB
    "pitch": 4.0,  # Hz
    "pitch_range": 0.5,  # semitones
    "pause": 0.02,  # ratio
}


@dataclass
class _Running:
    """Windowed mean and standard deviation for one dimension."""

    window: int
    floor: float
    """Lower bound on standard deviation. See `_VARIANCE_FLOOR`."""

    _values: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self._values = deque(maxlen=self.window)

    def add(self, value: float) -> None:
        self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    @property
    def stdev(self) -> float:
        n = len(self._values)
        if n < 2:
            return 0.0
        mu = self.mean
        variance = sum((v - mu) ** 2 for v in self._values) / (n - 1)
        return math.sqrt(variance)

    def z(self, value: float) -> float:
        """Standard score against a floored standard deviation, clamped.

        Two guards, for opposite failure modes:

        - **Flooring** stops a very consistent speaker from having every deviation
          reported as zero. Without it, the more regular someone is, the blinder we
          become to them changing.
        - **Clamping** stops one cough from producing a 40-sigma spike that swings the
          whole policy off a single artifact.
        """
        sd = max(self.stdev, self.floor)
        return max(-4.0, min(4.0, (value - self.mean) / sd))


class SpeakerBaseline:
    """Tracks one speaker's normal, and scores turns against it.

    Not an identity mechanism. Callers supply the key (a device certificate,
    per docs/DESIGN.md §3); this class never tries to work out who is speaking.
    """

    __slots__ = ("_energy", "_pause", "_pitch", "_pitch_range", "_rate", "_window")

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        if window < MIN_SAMPLES:
            raise ValueError(f"window must be >= {MIN_SAMPLES}, got {window}")
        self._window = window
        self._rate = _Running(window, _VARIANCE_FLOOR["rate"])
        self._energy = _Running(window, _VARIANCE_FLOOR["energy"])
        self._pitch = _Running(window, _VARIANCE_FLOOR["pitch"])
        self._pitch_range = _Running(window, _VARIANCE_FLOOR["pitch_range"])
        self._pause = _Running(window, _VARIANCE_FLOOR["pause"])

    @property
    def sample_count(self) -> int:
        return self._rate.count

    @property
    def is_warm(self) -> bool:
        """True once there is enough history to produce a meaningful delta."""
        return self.sample_count >= MIN_SAMPLES

    def observe(self, frame: ProsodyFrame) -> None:
        """Fold one utterance into the baseline."""
        self._rate.add(frame.speaking_rate)
        self._energy.add(frame.energy_db)
        self._pitch.add(frame.pitch_hz)
        self._pitch_range.add(frame.pitch_range_semitones)
        self._pause.add(frame.pause_ratio)

    def compare(self, frame: ProsodyFrame) -> ProsodyDelta | None:
        """Score a frame against the baseline, or None if still cold.

        Returning None rather than a zero-delta is deliberate: callers must
        distinguish "normal for them" from "we don't know them yet", because the
        correct behaviour differs (act vs. stay neutral and probe).
        """
        if not self.is_warm:
            return None
        return ProsodyDelta(
            rate_z=self._rate.z(frame.speaking_rate),
            energy_z=self._energy.z(frame.energy_db),
            pitch_z=self._pitch.z(frame.pitch_hz),
            pitch_range_z=self._pitch_range.z(frame.pitch_range_semitones),
            pause_z=self._pause.z(frame.pause_ratio),
        )

    def observe_and_compare(self, frame: ProsodyFrame) -> ProsodyDelta | None:
        """Score against history, *then* fold in.

        Order matters. Scoring a frame against a baseline that already contains it
        dilutes exactly the signal we are trying to detect — a genuinely unusual
        turn would partly normalise itself away.
        """
        delta = self.compare(frame)
        self.observe(frame)
        return delta
