"""The negotiator — holding the floor while the server thinks.

This is the component most likely to make Aura unpleasant, so it is built around
restraint rather than capability. The reference failure is a major assistant launch
that shipped backchannelling and was described by reviewers as interrupting people and
laughing at things that weren't jokes (docs/DESIGN.md §9).

Three rules, all enforced here:

1. **Content-free.** Acknowledgements commit to nothing. If the server comes back with
   something different, an "mm-hm" is never contradicted — a guess at the answer would be.
2. **Rate-limited.** A backchannel every turn is worse than none. Frequency is capped and
   decays with use.
3. **Silence is the default.** `consider()` returns None far more often than not, and
   that is the correct behaviour rather than a failure to produce output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

MIN_GAP_SECONDS = 4.0
"""Minimum spacing between acknowledgements. Below this it reads as a tic."""

MAX_RATE = 0.30
"""Ceiling on the fraction of turns that get one. Human backchannel rates vary widely;
this sits at the low end deliberately, because over-acknowledging is far more annoying
than under-acknowledging."""

THINKING_THRESHOLD_SECONDS = 0.9
"""How long the server may take before the silence needs covering. Below this the gap
reads as ordinary conversational timing."""


class Acknowledgement(Enum):
    """The complete vocabulary. Deliberately tiny and deliberately empty of meaning.

    Nothing here can be wrong, because nothing here asserts anything.
    """

    CONTINUER = "mm-hm"
    """Keep going — the most neutral thing available."""

    RECEIPT = "okay"
    """Heard you. Slightly stronger, still commits to nothing."""

    THINKING = "let me think"
    """Explicit: something is being worked on. Honest about the delay rather than
    pretending it isn't happening."""


@dataclass
class NegotiatorState:
    """What the negotiator remembers between turns."""

    turns_seen: int = 0
    acknowledgements: int = 0
    seconds_since_last: float = field(default=MIN_GAP_SECONDS * 2)
    """Starts above the gap so the first turn is eligible."""

    @property
    def rate(self) -> float:
        return self.acknowledgements / self.turns_seen if self.turns_seen else 0.0


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether to say something, and why not when not.

    The reason matters: "did nothing" is the common path, and without a recorded reason
    an over-quiet or over-chatty negotiator is very hard to diagnose.
    """

    acknowledgement: Acknowledgement | None
    reason: str

    @property
    def should_speak(self) -> bool:
        return self.acknowledgement is not None


class Negotiator:
    """Decides whether to fill a gap out loud.

    Deterministic when seeded, so behaviour is reproducible in tests and in evaluation.
    """

    __slots__ = ("_max_rate", "_min_gap", "_random", "_state")

    def __init__(
        self,
        *,
        min_gap_seconds: float = MIN_GAP_SECONDS,
        max_rate: float = MAX_RATE,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= max_rate <= 1.0:
            raise ValueError(f"max_rate must be in [0,1], got {max_rate}")
        if min_gap_seconds < 0:
            raise ValueError(f"min_gap_seconds must be non-negative, got {min_gap_seconds}")
        self._state = NegotiatorState()
        self._random = random.Random(seed)
        self._min_gap = min_gap_seconds
        self._max_rate = max_rate

    @property
    def state(self) -> NegotiatorState:
        return self._state

    def observe_turn(self, seconds_since_last_ack: float) -> None:
        """Note that the user finished a turn."""
        self._state.turns_seen += 1
        self._state.seconds_since_last = seconds_since_last_ack

    def consider(
        self,
        *,
        expected_wait_seconds: float,
        user_still_speaking: bool = False,
        profile_allows: bool = True,
    ) -> Decision:
        """Decide whether to acknowledge.

        Checks run cheapest-and-most-absolute first, so the common answer — no — is
        reached without evaluating anything expensive.
        """
        if user_still_speaking:
            # The single worst failure. Nothing overrides this.
            return Decision(None, "user is still speaking")

        if not profile_allows:
            return Decision(None, "this person has asked for no backchannelling")

        if expected_wait_seconds < THINKING_THRESHOLD_SECONDS:
            return Decision(None, f"answer is {expected_wait_seconds:.1f}s away — no gap to cover")

        if self._state.seconds_since_last < self._min_gap:
            return Decision(
                None, f"acknowledged {self._state.seconds_since_last:.1f}s ago — too soon"
            )

        if self._state.rate >= self._max_rate and self._state.turns_seen >= 3:
            return Decision(None, f"already acknowledging {self._state.rate:.0%} of turns")

        choice = self._choose(expected_wait_seconds)
        self._state.acknowledgements += 1
        self._state.seconds_since_last = 0.0
        return Decision(choice, f"covering a {expected_wait_seconds:.1f}s gap")

    def _choose(self, wait_seconds: float) -> Acknowledgement:
        """Pick which token to use.

        A long wait gets the honest one. Being told "let me think" is better than an
        unexplained silence; being told it for a half-second gap is theatre.
        """
        if wait_seconds >= 2.5:
            return Acknowledgement.THINKING
        return self._random.choice([Acknowledgement.CONTINUER, Acknowledgement.RECEIPT])

    def reset(self) -> None:
        self._state = NegotiatorState()
