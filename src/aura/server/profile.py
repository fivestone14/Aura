"""The Interpreter — what we have learned about how one person likes to be talked to.

Two hard rules, both enforced here rather than by convention:

1. **Delivery only, never substance** (docs/DESIGN.md §5). The schema is closed. There
   is nowhere to record an opinion, so an opinion cannot leak into delivery logic.
2. **Rates, not incidents** (docs/DESIGN.md §4). One irritated reply is noise. A trait
   needs repetition and carries its sample count, so callers can weigh it.

Everything is plain data and round-trips to a readable dict — the user is entitled to
read, correct, and delete this.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

MIN_OBSERVATIONS_TO_APPLY = 8
"""Below this an observed rate is not acted on. Chosen so a bad afternoon cannot
rewrite someone's profile, and revisited once real usage data exists."""


class Verbosity(Enum):
    TERSE = "terse"
    NORMAL = "normal"
    DETAILED = "detailed"


class Pace(Enum):
    SLOW = "slow"
    NORMAL = "normal"
    QUICK = "quick"


# The closed set of things a profile may contain. Adding to this list is a design
# decision, not a detail: every entry must describe *how to say something*, never
# *what to say*. A field like `agrees_with_x` or `preferred_conclusion` is rejected.
_ALLOWED_PREFERENCE_KEYS = frozenset(
    {"verbosity", "pace", "formality", "wants_preamble", "tolerates_backchannel", "wants_probes"}
)


class SubstanceLeakError(ValueError):
    """Raised when something tries to store content-shaped data in the profile.

    This is the delivery/substance barrier failing closed. See docs/DESIGN.md §5.
    """


@dataclass(frozen=True, slots=True)
class ObservedRate:
    """How often something happens, and how sure we are.

    Storing the denominator is the point. `0.24` alone is a claim; `0.24 (n=31)` is
    evidence, and a caller can tell the difference between a pattern and a fluke.
    """

    hits: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def is_actionable(self) -> bool:
        return self.total >= MIN_OBSERVATIONS_TO_APPLY

    def record(self, occurred: bool) -> ObservedRate:
        return ObservedRate(hits=self.hits + int(occurred), total=self.total + 1)

    def __str__(self) -> str:
        return f"{self.rate:.2f} (n={self.total})"


@dataclass
class Profile:
    """One person's delivery preferences, scoped by context.

    `context` exists because preferences are task-specific — the same person wants a
    one-liner for a lookup and depth for a decision (docs/DESIGN.md §4). A single
    global profile would average those into something wrong for both.
    """

    key: str
    """Opaque caller-supplied identifier — a device certificate fingerprint, not a
    person. This class never derives it from audio."""

    context: str = "default"

    verbosity: Verbosity = Verbosity.NORMAL
    pace: Pace = Pace.NORMAL
    wants_preamble: bool = True
    tolerates_backchannel: bool = True
    wants_probes: bool = True

    interrupted_long_answers: ObservedRate = field(default_factory=ObservedRate)
    """Rises when the user talks over a long reply — the strongest implicit signal
    that they want brevity."""

    corrections: list[str] = field(default_factory=list)
    """Verbatim standing corrections the user gave. Gold-standard, and rare."""

    def apply_correction(self, text: str) -> None:
        """Record an explicit instruction about delivery.

        Rejects anything that reads as a claim about the world rather than about how
        to speak. The check is intentionally crude and errs toward rejection: a
        wrongly-rejected correction is a minor annoyance, whereas a stored opinion is
        a silent bias in every future turn.
        """
        lowered = text.lower().strip()
        if not lowered:
            raise ValueError("correction cannot be empty")
        _reject_if_substance(lowered)
        self.corrections.append(text.strip())

    def record_interruption(self, *, reply_was_long: bool, interrupted: bool) -> None:
        """Fold one observation about long replies into the rate."""
        if reply_was_long:
            self.interrupted_long_answers = self.interrupted_long_answers.record(interrupted)

    @property
    def effective_verbosity(self) -> Verbosity:
        """Declared verbosity, overridden by behaviour once behaviour is convincing.

        Observed behaviour beats stated preference here — not because people lie, but
        because they answer a settings question once and then reveal the truth over
        thirty turns. Only applies past the sample-count floor.
        """
        rate = self.interrupted_long_answers
        if rate.is_actionable and rate.rate >= 0.25:
            return Verbosity.TERSE
        return self.verbosity

    def to_dict(self) -> dict[str, Any]:
        """Readable export. This is what the user sees when they ask what we know."""
        data = asdict(self)
        data["verbosity"] = self.verbosity.value
        data["pace"] = self.pace.value
        data["effective_verbosity"] = self.effective_verbosity.value
        data["interrupted_long_answers"] = str(self.interrupted_long_answers)
        return data


# Words that indicate a claim about the world rather than about delivery. Deliberately
# short: a long list invites false positives on ordinary delivery feedback.
_SUBSTANCE_MARKERS = (
    "always agree",
    "agree with me",
    "tell me i'm right",
    "never disagree",
    "don't correct me",
    "say i'm right",
    "take my side",
)


def _reject_if_substance(lowered: str) -> None:
    for marker in _SUBSTANCE_MARKERS:
        if marker in lowered:
            raise SubstanceLeakError(
                f"refusing to store {marker!r}: the profile governs delivery, not substance "
                "(docs/DESIGN.md §5)"
            )


def validate_preference_keys(keys: object) -> None:
    """Guard for any path that builds a profile from untrusted input.

    The schema is the enforcement mechanism, so anything constructing one from
    outside — config, an API call, a restored file — passes through here first.
    """
    if not isinstance(keys, (set, frozenset, list, tuple, dict)):
        raise TypeError(f"expected a collection of keys, got {type(keys).__name__}")
    unknown = set(keys) - _ALLOWED_PREFERENCE_KEYS
    if unknown:
        raise SubstanceLeakError(
            f"unknown profile keys {sorted(unknown)}; the schema is closed to keep the "
            "profile delivery-only (docs/DESIGN.md §5)"
        )
