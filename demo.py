"""A conversation with no credentials, no network, and no microphone.

Run it to see the loop work end to end:

    .venv/bin/python demo.py
"""

from __future__ import annotations

import asyncio

from aura.server import Chair, Council, EchoBackend, LocalService
from aura.session import Session
from aura.types import ProsodyFrame


async def main() -> None:
    service = LocalService(Council(EchoBackend()), Chair())
    session = Session(session_key="demo-device", transport=service)

    # Ten ordinary turns, so the baseline learns what normal sounds like.
    calm = ProsodyFrame(4.0, -20.0, 120.0, 6.0, 0.25)
    for _ in range(10):
        await session.handle_turn("just chatting", calm)

    print(f"baseline warm after {session.baseline.sample_count} turns\n")

    for label, frame in (
        ("calm, as usual", calm),
        ("fast and loud", ProsodyFrame(8.5, -8.0, 190.0, 7.0, 0.02)),
        ("flat and quiet", ProsodyFrame(3.2, -30.0, 105.0, 1.2, 0.45)),
    ):
        reply = await session.handle_turn("what should I do", frame)
        print(f"{label}")
        print(f"  register  {reply.prosody.register.value}")
        print(f"  rate      {reply.prosody.rate_scale:.2f}x")
        print(f"  pitch     {reply.prosody.pitch_shift_semitones:+.1f} semitones")
        print(f"  why       {reply.rationale.split(' | ')[0]}")
        print()

    print(f"turns={session.stats.turns} fallbacks={session.stats.fallbacks}")


if __name__ == "__main__":
    asyncio.run(main())
