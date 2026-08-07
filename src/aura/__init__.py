"""Aura — a prosody-aware conversational voice pipeline.

Two halves, deliberately separated (docs/DESIGN.md §7):

    aura.client   runs on the device. Real-time, offline-capable, no credentials.
    aura.server   runs on a host you control. Does the thinking.

First principle: stop trying to guess who someone is; get better at noticing what
they do.
"""

from aura.session import Session, SpokenReply
from aura.transport import (
    ThinkRequest,
    ThinkResponse,
    Transport,
    UnavailableTransport,
)
from aura.types import (
    EGEMAPS_FEATURE_COUNT,
    ProsodyDelta,
    ProsodyFrame,
    ProsodyTarget,
    Register,
    Turn,
)

__version__ = "0.1.0"

__all__ = [
    "EGEMAPS_FEATURE_COUNT",
    "ProsodyDelta",
    "ProsodyFrame",
    "ProsodyTarget",
    "Register",
    "Session",
    "SpokenReply",
    "ThinkRequest",
    "ThinkResponse",
    "Transport",
    "Turn",
    "UnavailableTransport",
    "__version__",
]
