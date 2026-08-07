"""Prosody policy — choosing how to say the reply.

Counter-regulation: when someone is activated, answer *below* them rather than
matching. Slower, quieter, lower. Grounded in de-escalation practice and
infant-directed speech, where soothing is a low falling contour and explicitly not a
mirror of the infant's distress (docs/DESIGN.md §1).

This module is deliberately rules, not a model. There is no corpus that labels "given
this input prosody, the ideal reply prosody was X", so a learned policy here would be
fitting noise. Rules are also inspectable when the output is wrong, which a model
would not be.
"""

from __future__ import annotations

from dataclasses import dataclass

from aura.server.profile import Pace, Profile, Verbosity
from aura.types import ProsodyDelta, ProsodyTarget, Register

ACTIVATION_THRESHOLD = 0.75
"""Standard deviations above a speaker's own norm before counter-regulation engages.
Below this, mirroring slightly is natural rapport and shouldn't be suppressed."""

STRONG_ACTIVATION = 1.75
"""Where a firmer counter-regulation applies. Not a claim that the person is
distressed — only that they are well outside their own normal range."""

FLATNESS_THRESHOLD = -1.0
"""Pitch range this far below normal reads as withdrawn or exhausted. The correct
response is warmth, not calm — calm on top of flat is just cold."""


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A target plus the reason for it.

    The rationale is not decoration: it is what makes a wrong decision debuggable,
    and what the evaluation harness scores against.
    """

    target: ProsodyTarget
    rationale: str


def decide(delta: ProsodyDelta | None, profile: Profile | None = None) -> PolicyDecision:
    """Choose delivery for one reply.

    `delta` is None when the speaker has no baseline yet. That is a real state, not an
    error, and it has its own correct behaviour: stay neutral rather than guess.
    """
    if delta is None:
        return PolicyDecision(
            target=_apply_profile(ProsodyTarget(register=Register.NEUTRAL), profile),
            rationale="no baseline yet — staying neutral rather than assuming",
        )

    activation = delta.activation

    if delta.pitch_range_z <= FLATNESS_THRESHOLD and activation < ACTIVATION_THRESHOLD:
        base = ProsodyTarget(
            rate_scale=0.95,
            pitch_shift_semitones=0.5,
            energy_scale=1.0,
            pause_scale=1.1,
            register=Register.WARM,
        )
        rationale = f"flat delivery (pitch range {delta.pitch_range_z:+.1f}σ) — warmth, not calm"

    elif activation >= ACTIVATION_THRESHOLD:
        # Interpolate rather than snapping between fixed levels. A step function would
        # make someone hovering near a threshold hear the register flip back and forth
        # between turns, which is more unsettling than either setting on its own.
        intensity = _ramp(activation, ACTIVATION_THRESHOLD, STRONG_ACTIVATION)
        base = ProsodyTarget(
            rate_scale=_lerp(0.94, 0.80, intensity),
            pitch_shift_semitones=_lerp(-0.5, -2.0, intensity),
            energy_scale=_lerp(0.95, 0.80, intensity),
            pause_scale=_lerp(1.1, 1.4, intensity),
            register=Register.CALM,
        )
        descriptor = "strongly activated" if intensity >= 0.5 else "activated"
        rationale = f"{descriptor} ({activation:+.1f}σ) — counter-regulating at {intensity:.0%}"

    else:
        base = ProsodyTarget(register=Register.NEUTRAL)
        rationale = f"within their normal range ({activation:+.1f}σ) — no adjustment"

    return PolicyDecision(target=_apply_profile(base, profile), rationale=rationale)


def _apply_profile(target: ProsodyTarget, profile: Profile | None) -> ProsodyTarget:
    """Let learned preferences adjust delivery — within limits.

    A person who likes things quick still gets slowed down when they're wound up; the
    profile shifts the result, it doesn't override the policy. Hence multiplying into
    the existing scale rather than replacing it.
    """
    if profile is None:
        return target

    rate = target.rate_scale
    if profile.pace is Pace.QUICK:
        rate *= 1.10
    elif profile.pace is Pace.SLOW:
        rate *= 0.92

    if profile.effective_verbosity is Verbosity.TERSE:
        # Terse speech is not just shorter — it carries less trailing silence.
        pause = target.pause_scale * 0.9
    else:
        pause = target.pause_scale

    return ProsodyTarget(
        rate_scale=_clamp(rate, 0.5, 2.0),
        pitch_shift_semitones=target.pitch_shift_semitones,
        energy_scale=target.energy_scale,
        pause_scale=pause,
        register=target.register,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ramp(value: float, start: float, end: float) -> float:
    """Position of `value` between `start` and `end`, as 0.0-1.0.

    Saturates at both ends: past `end` the response stops intensifying, because
    counter-regulating harder than "slow and quiet" starts to read as condescension.
    """
    if end <= start:
        return 1.0
    return _clamp((value - start) / (end - start), 0.0, 1.0)


def _lerp(low: float, high: float, t: float) -> float:
    return low + (high - low) * t
