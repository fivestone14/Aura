"""Client side — everything that must run on the user's device.

These components have hard real-time budgets and cannot tolerate a network hop:
echo cancellation needs the speaker signal as it plays, prosody and turn-taking run
on every audio frame, and an acknowledgement has to land in roughly 200 ms
(docs/DESIGN.md §7).

Nothing here requires credentials or a network connection. If the server is
unreachable, this half still works — degraded, not dead.
"""

from aura.client.audio import (
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioLoop,
    EchoCanceller,
    Frame,
    PassthroughCanceller,
    ReferenceTap,
    SubtractiveCanceller,
    UpstreamCanceller,
)
from aura.client.baseline import MIN_SAMPLES, SpeakerBaseline
from aura.client.negotiator import Acknowledgement, Decision, Negotiator
from aura.client.policy import PolicyDecision, decide

__all__ = [
    "FRAME_MS",
    "FRAME_SAMPLES",
    "MIN_SAMPLES",
    "SAMPLE_RATE",
    "Acknowledgement",
    "AudioLoop",
    "Decision",
    "EchoCanceller",
    "Frame",
    "Negotiator",
    "PassthroughCanceller",
    "PolicyDecision",
    "ReferenceTap",
    "SpeakerBaseline",
    "SubtractiveCanceller",
    "UpstreamCanceller",
    "decide",
]
