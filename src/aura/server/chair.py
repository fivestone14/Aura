"""The chair — turns several readings into one thing to say.

Two jobs, and the second is easy to underrate:

1. **Merge.** Take what the council produced and commit to a single reply.
2. **Enforce the spoken register.** Text destined for a speaker has different rules
   from text destined for a screen: no markdown, no lists, short sentences, no
   preamble. A model asked for prose will happily emit a bulleted list, and a bulleted
   list read aloud is unbearable.

The prosody plan is emitted **after** the text, never before. Under streaming
aggregation the chair starts composing from partial council output, so a plan chosen up
front would be committing to a tone for content that has not arrived yet
(docs/DESIGN.md §5, §9).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aura.server.council import Contribution, CouncilResult, Role
from aura.types import ProsodyTarget

MAX_SPOKEN_SECONDS = 10.0
"""Comprehension of continuous speech falls off past roughly this long. Not a hard
truncation point — a signal that the reply is doing too much at once."""

WORDS_PER_SECOND = 2.8
"""Conversational speaking rate, for estimating how long a reply will take to say."""

MAX_SENTENCE_WORDS = 22
"""Past this a spoken sentence is hard to follow, because the listener cannot re-read."""


@dataclass(frozen=True, slots=True)
class Reply:
    """What to say, and how.

    Field order mirrors emission order: text is settled before tone is chosen.
    """

    text: str
    prosody: ProsodyTarget
    rationale: str

    @property
    def estimated_seconds(self) -> float:
        return len(self.text.split()) / WORDS_PER_SECOND

    @property
    def is_overlong(self) -> bool:
        return self.estimated_seconds > MAX_SPOKEN_SECONDS


# Markdown and other screen-only formatting. Read aloud, a bullet is a pause and an
# asterisk is nothing at all, so these have to go before synthesis.
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_HEADING = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*{1,2}|_{1,2}|`+)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_WHITESPACE = re.compile(r"\s+")

# Openings that delay the answer. Spoken, these are worse than on screen: the listener
# cannot skim past them.
_PREAMBLE = re.compile(
    r"^\s*(?:"
    r"(?:well|so|okay|ok|sure|certainly|absolutely|of course|great question)[,!.]?\s+"
    # Contractions matter here: models write "I'd be happy to", not "I would be happy to".
    r"|i(?:'d|'ll| will| can| would)\b[^.!?]*[.!?]\s+"
    r"|let me\b[^.!?]*[.!?]\s+"
    r"|here(?:'s| is)\b[^.!?]*[:.]\s+"
    r")",
    re.IGNORECASE,
)


def make_speakable(text: str) -> str:
    """Strip anything that only makes sense on a screen.

    Applied to whatever the council returns rather than trusting the prompt. Prompts
    are guidance; this is a guarantee.
    """
    text = _LINK.sub(r"\1", text)
    text = _HEADING.sub("", text)
    # A bullet becomes a sentence boundary — that is what the list was standing in for.
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()

    # Strip preambles repeatedly: models stack them ("Sure! Let me explain. Here's...").
    stripped_any = False
    for _ in range(3):
        stripped = _PREAMBLE.sub("", text, count=1)
        if stripped == text:
            break
        text, stripped_any = stripped, True

    text = text.strip()

    # Removing "Well, " leaves "the answer is 42." — grammatically wrong, and speech
    # synthesis uses sentence casing as a prosodic cue for where a sentence begins.
    if stripped_any and text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def find_long_sentences(text: str) -> list[str]:
    """Sentences over the spoken length limit. Reported, not silently rewritten —
    splitting prose mechanically produces worse output than leaving it long."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if len(s.split()) > MAX_SENTENCE_WORDS]


class Chair:
    """Merges council contributions into a single reply."""

    def __init__(self, *, enforce_speakable: bool = True) -> None:
        self._enforce = enforce_speakable

    def merge(self, result: CouncilResult, prosody: ProsodyTarget) -> Reply | None:
        """Produce one reply from whatever the council returned.

        Returns None on an empty council. That is a real state — every member timed out
        or failed — and the caller must handle it, because saying nothing is better than
        saying something invented to fill the gap.
        """
        if result.is_empty:
            return None

        primary = self._select_primary(result.contributions)
        text = primary.text
        if self._enforce:
            text = make_speakable(text)

        if not text:
            # Formatting stripping emptied it — the member returned only markup.
            return None

        return Reply(
            text=text,
            prosody=prosody,
            rationale=self._explain(result, primary),
        )

    def _select_primary(self, contributions: list[Contribution]) -> Contribution:
        """Choose which reading leads.

        Intent leads when present: the transcript is usually partial, and answering what
        someone meant beats answering a garbled literal reading. The skeptic never leads
        — its role is to check, and a reply that opens with an objection is hostile even
        when the objection is right.
        """
        by_role = {c.role: c for c in contributions}
        for role in (Role.INTENT, Role.LITERAL):
            if role in by_role:
                return by_role[role]
        return contributions[0]

    def _explain(self, result: CouncilResult, primary: Contribution) -> str:
        parts = [f"led with {primary.role.value}"]
        if result.timed_out:
            missing = ", ".join(r.value for r in result.timed_out)
            parts.append(f"missing {missing}")
        if len(result.contributions) == 1:
            parts.append("single reading only")
        return "; ".join(parts)
