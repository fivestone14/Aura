"""Core value types.

Everything here is immutable. A turn's measurements are facts about a moment that
already happened; nothing downstream should be able to rewrite them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# The eGeMAPS v02 functional set. Named here so the vector's shape is a contract
# rather than an assumption held in three places.
EGEMAPS_FEATURE_COUNT = 88


class Register(Enum):
    """How a reply should be delivered. Delivery only — never content.

    See docs/DESIGN.md §5. Adding a member that implies a *stance* rather than a
    *manner* breaks the delivery/substance separation.
    """

    CALM = "calm"
    NEUTRAL = "neutral"
    WARM = "warm"
    BRISK = "brisk"


@dataclass(frozen=True, slots=True)
class ProsodyFrame:
    """Raw acoustics for one utterance, in absolute units.

    Absolute values are only meaningful relative to a speaker's own history, which
    is what `SpeakerBaseline` is for. Policy never reads this directly.
    """

    speaking_rate: float
    """Syllables per second."""

    energy_db: float
    """RMS intensity in dBFS."""

    pitch_hz: float
    """Median f0 over voiced regions."""

    pitch_range_semitones: float
    """Spread of f0; low values read as flat or withdrawn."""

    pause_ratio: float
    """Fraction of the utterance that is silence. 0.0-1.0."""

    features: tuple[float, ...] = field(default=())
    """Full eGeMAPS vector, when available. Empty is valid — the five named fields
    above are what policy uses; this is for evaluation and future models."""

    def __post_init__(self) -> None:
        if self.features and len(self.features) != EGEMAPS_FEATURE_COUNT:
            raise ValueError(
                f"expected {EGEMAPS_FEATURE_COUNT} eGeMAPS features, got {len(self.features)}"
            )
        if not 0.0 <= self.pause_ratio <= 1.0:
            raise ValueError(f"pause_ratio must be in [0,1], got {self.pause_ratio}")


@dataclass(frozen=True, slots=True)
class ProsodyDelta:
    """How this utterance compares to the speaker's own norm.

    This — not `ProsodyFrame` — is what policy acts on. "Fast" means fast *for them*.
    Values are z-scores: +1.0 is one standard deviation above their baseline.
    """

    rate_z: float
    energy_z: float
    pitch_z: float
    pitch_range_z: float
    pause_z: float

    @property
    def activation(self) -> float:
        """A single scalar for how wound-up this sounds, relative to their norm.

        Deliberately *not* called `arousal`: this is a weighted sum of three measured
        deviations, not an inference about an emotional state. See docs/DESIGN.md §2.

        Faster, louder, higher-pitched all push up. Pausing pushes down.
        """
        return (self.rate_z + self.energy_z + self.pitch_z) / 3.0 - (self.pause_z * 0.5)


@dataclass(frozen=True, slots=True)
class ProsodyTarget:
    """What the renderer is asked to produce.

    Multipliers and offsets rather than absolutes, so the same target means the same
    thing for any voice.
    """

    rate_scale: float = 1.0
    """1.0 = the voice's natural rate. 0.85 = 15% slower."""

    pitch_shift_semitones: float = 0.0
    energy_scale: float = 1.0
    pause_scale: float = 1.0
    register: Register = Register.NEUTRAL

    def __post_init__(self) -> None:
        # Bounds are perceptual, not technical: outside these the voice stops
        # sounding like a person and starts sounding like a broken tape.
        if not 0.5 <= self.rate_scale <= 2.0:
            raise ValueError(f"rate_scale out of range: {self.rate_scale}")
        if not -12.0 <= self.pitch_shift_semitones <= 12.0:
            raise ValueError(f"pitch_shift out of range: {self.pitch_shift_semitones}")
        if not 0.25 <= self.energy_scale <= 2.0:
            raise ValueError(f"energy_scale out of range: {self.energy_scale}")


@dataclass(frozen=True, slots=True)
class Turn:
    """One thing the user said, with how they said it."""

    text: str
    prosody: ProsodyFrame
    delta: ProsodyDelta | None = None
    """None until the speaker has enough history for a baseline."""

    is_partial: bool = False
    """True for speculative turns fired before the user finished. See docs/DESIGN.md §9."""
