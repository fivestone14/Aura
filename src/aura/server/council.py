"""The council — several readings of one turn, merged into a single reply.

Two structural constraints, both load-bearing:

1. **The skeptic is blind to the profile** (docs/DESIGN.md §5b). Wiring the profile to
   every member looks tidier and silently destroys the one voice whose job is to
   disagree, because it inherits the bias it exists to catch. The type system enforces
   this: `Skeptic` cannot receive a profile.
2. **Parallel, deadline-bounded, never a debate loop** (docs/DESIGN.md §8). Debate is
   seconds-scale and unaffordable, and unguided debate underperforms a single careful
   pass anyway.

The council is justified by robustness on *noisy* input, not by raw intelligence. On
clean transcripts a single member matches it — so `CouncilConfig.members` is
deliberately easy to reduce to one when the A/B says so.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from aura.server.profile import Profile
from aura.types import Turn

DEFAULT_DEADLINE_SECONDS = 2.0
"""Hard cutoff. Past roughly this point the conversation has already gone quiet too
long, and a late-but-better answer is worse than a timely adequate one."""


class Role(Enum):
    LITERAL = "literal"
    """Takes the words at face value."""

    INTENT = "intent"
    """Reads what they probably meant — useful when the transcript is fragmentary."""

    SKEPTIC = "skeptic"
    """Looks for what is missing or wrong. Never sees the profile."""


@dataclass(frozen=True, slots=True)
class Contribution:
    role: Role
    text: str
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


class Member(ABC):
    """One reading of the turn."""

    role: Role

    @abstractmethod
    async def contribute(self, turn: Turn, profile: Profile | None) -> Contribution:
        """Produce a reading.

        `profile` is None for roles that must not see it. Implementations should not
        assume it is present.
        """


class Skeptic(Member):
    """Resists consensus.

    Takes no profile — not as a convention, but because `contribute` discards it. If a
    future refactor makes this class profile-aware, the blinding is gone and §5b is
    violated; the test suite asserts against exactly that.
    """

    role = Role.SKEPTIC

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    async def contribute(self, turn: Turn, profile: Profile | None) -> Contribution:
        del profile  # deliberate: the skeptic is blind by design
        text = await self._backend.generate(
            role=Role.SKEPTIC, turn=turn, profile=None
        )
        return Contribution(role=Role.SKEPTIC, text=text)


class StandardMember(Member):
    """A profile-aware reading."""

    def __init__(self, role: Role, backend: Backend) -> None:
        if role is Role.SKEPTIC:
            raise ValueError("use Skeptic for the skeptic role — it must not receive a profile")
        self.role = role
        self._backend = backend

    async def contribute(self, turn: Turn, profile: Profile | None) -> Contribution:
        text = await self._backend.generate(role=self.role, turn=turn, profile=profile)
        return Contribution(role=self.role, text=text)


class Backend(ABC):
    """Whatever actually produces text. Kept abstract so the council's structure can be
    tested without a model, and so the model can be swapped without touching this file."""

    @abstractmethod
    async def generate(self, *, role: Role, turn: Turn, profile: Profile | None) -> str: ...


@dataclass(frozen=True, slots=True)
class CouncilConfig:
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS
    roles: tuple[Role, ...] = (Role.LITERAL, Role.INTENT, Role.SKEPTIC)

    def __post_init__(self) -> None:
        if self.deadline_seconds <= 0:
            raise ValueError("deadline must be positive")
        if not self.roles:
            raise ValueError("council needs at least one role")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError(f"duplicate roles: {self.roles}")


@dataclass
class CouncilResult:
    contributions: list[Contribution] = field(default_factory=list)
    timed_out: list[Role] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.contributions


class Council:
    """Runs members concurrently against a deadline."""

    def __init__(self, backend: Backend, config: CouncilConfig | None = None) -> None:
        self._config = config or CouncilConfig()
        self._members: list[Member] = [
            Skeptic(backend) if role is Role.SKEPTIC else StandardMember(role, backend)
            for role in self._config.roles
        ]

    async def deliberate(self, turn: Turn, profile: Profile | None = None) -> CouncilResult:
        """Gather readings, discarding any that miss the deadline.

        A slow member is dropped rather than waited for. Partial input is the normal
        case, so a partial council is an acceptable one — the chair works with what
        arrived.
        """
        tasks = {
            asyncio.create_task(member.contribute(turn, profile)): member.role
            for member in self._members
        }

        done, pending = await asyncio.wait(tasks, timeout=self._config.deadline_seconds)

        result = CouncilResult()
        for task in pending:
            result.timed_out.append(tasks[task])
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            error = task.exception()
            if error is not None:
                # A failed member is a degraded council, not a failed turn.
                result.timed_out.append(tasks[task])
                continue
            result.contributions.append(task.result())

        result.contributions.sort(key=lambda c: self._config.roles.index(c.role))
        return result
