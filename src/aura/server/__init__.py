"""Server side — the reasoning, which is allowed to be slow.

Runs on a host the operator controls. Reached over an encrypted tunnel; only text
and prosody deltas cross it, never audio (docs/DESIGN.md §7).
"""

from aura.server.backends import ClaudeBackend, EchoBackend
from aura.server.chair import Chair, Reply, make_speakable
from aura.server.config import CredentialError, Settings, resolve_api_key
from aura.server.council import Council, CouncilConfig, CouncilResult, Role
from aura.server.profile import Pace, Profile, SubstanceLeakError, Verbosity
from aura.server.service import LocalService
from aura.server.store import JsonStore, MemoryStore, ProfileStore

__all__ = [
    "Chair",
    "ClaudeBackend",
    "Council",
    "CouncilConfig",
    "CouncilResult",
    "CredentialError",
    "EchoBackend",
    "JsonStore",
    "LocalService",
    "MemoryStore",
    "Pace",
    "Profile",
    "ProfileStore",
    "Reply",
    "Role",
    "Settings",
    "SubstanceLeakError",
    "Verbosity",
    "make_speakable",
    "resolve_api_key",
]
