"""Where profiles live between conversations.

A profile that vanishes on restart is not a profile — the whole premise is that Aura
gets better at talking to someone over weeks, which requires surviving a reboot.

Two implementations, deliberately:

- `MemoryStore` for tests and for a session that should leave no trace.
- `JsonStore` for real use. Plain, readable JSON on disk.

**JSON rather than an encrypted database, for now.** The design calls for SQLCipher with
a Secure-Enclave-wrapped key, and that is still right for production. But the profile is
meant to be something a person can open, read, correct, and delete, and a file they can
`cat` makes that true today rather than promised. Encryption is a swap behind this
interface, not a rewrite — which is why the interface exists.

⚠️ Contents are personal but not secret: stated preferences and aggregate speaking
statistics. No transcripts, no audio, no voice embeddings (docs/DESIGN.md §4). That is
what makes plaintext defensible in the interim; it would not be if this held recordings.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from aura.server.profile import ObservedRate, Pace, Profile, Verbosity

SCHEMA_VERSION = 1
"""Bumped when the on-disk shape changes. Stored with every record so an old file can be
recognised rather than misread as a new one with missing fields."""


class ProfileStore(ABC):
    """Load and save profiles by key."""

    @abstractmethod
    def load(self, key: str) -> Profile | None:
        """Return the stored profile, or None if this key is new."""

    @abstractmethod
    def save(self, profile: Profile) -> None:
        """Persist a profile, replacing any earlier version."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove a profile. Returns whether anything was there.

        Deletion must actually delete. "Forget me" that leaves a tombstone is not
        deletion, and this is the mechanism behind the user-facing promise.
        """

    @abstractmethod
    def keys(self) -> list[str]:
        """Every stored key. For the inspect-and-export surface."""

    def get_or_create(self, key: str) -> Profile:
        """Load, or hand back a fresh neutral profile.

        A new person gets neutral rather than an error or a guess — the cold-start
        behaviour the design calls for (docs/DESIGN.md §4).
        """
        return self.load(key) or Profile(key=key)


class MemoryStore(ProfileStore):
    """Keeps profiles in a dict. Nothing survives the process."""

    def __init__(self) -> None:
        self._profiles: dict[str, Profile] = {}

    def load(self, key: str) -> Profile | None:
        return self._profiles.get(key)

    def save(self, profile: Profile) -> None:
        self._profiles[profile.key] = profile

    def delete(self, key: str) -> bool:
        return self._profiles.pop(key, None) is not None

    def keys(self) -> list[str]:
        return sorted(self._profiles)


class JsonStore(ProfileStore):
    """One JSON file per profile, in a directory the operator owns."""

    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Owner-only. The contents are personal even though they are not secret.
        os.chmod(self._dir, 0o700)

    def _path(self, key: str) -> Path:
        return self._dir / f"{_safe_filename(key)}.json"

    def load(self, key: str) -> Profile | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            # A corrupt profile must not take down a conversation. Losing learned
            # preferences is recoverable; refusing to talk is not.
            return None
        if raw.get("schema_version") != SCHEMA_VERSION:
            return None
        return _from_dict(raw)

    def save(self, profile: Profile) -> None:
        """Write atomically.

        A half-written profile is worse than none — it reads as valid JSON with missing
        fields. Writing to a temporary file and renaming makes the update atomic on any
        POSIX filesystem, so a crash mid-write leaves the previous version intact.
        """
        path = self._path(profile.key)
        payload = _to_dict(profile)

        fd, tmp_name = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.json"))


def _to_dict(profile: Profile) -> dict[str, Any]:
    """Serialise. Written out field by field rather than reflectively, so adding a field
    to `Profile` cannot silently start persisting it — the delivery-only schema is a
    guarantee and should not be extended by accident (docs/DESIGN.md §5a)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "key": profile.key,
        "context": profile.context,
        "verbosity": profile.verbosity.value,
        "pace": profile.pace.value,
        "wants_preamble": profile.wants_preamble,
        "tolerates_backchannel": profile.tolerates_backchannel,
        "wants_probes": profile.wants_probes,
        "interrupted_long_answers": {
            "hits": profile.interrupted_long_answers.hits,
            "total": profile.interrupted_long_answers.total,
        },
        "corrections": list(profile.corrections),
    }


def _from_dict(raw: dict[str, Any]) -> Profile:
    """Deserialise, tolerating missing optional fields.

    Unknown keys are ignored rather than rejected: a file written by a newer version
    should degrade to the fields this version understands, not refuse to load.
    """
    rate = raw.get("interrupted_long_answers") or {}
    return Profile(
        key=raw["key"],
        context=raw.get("context", "default"),
        verbosity=Verbosity(raw.get("verbosity", "normal")),
        pace=Pace(raw.get("pace", "normal")),
        wants_preamble=bool(raw.get("wants_preamble", True)),
        tolerates_backchannel=bool(raw.get("tolerates_backchannel", True)),
        wants_probes=bool(raw.get("wants_probes", True)),
        interrupted_long_answers=ObservedRate(
            hits=int(rate.get("hits", 0)), total=int(rate.get("total", 0))
        ),
        corrections=list(raw.get("corrections", [])),
    )


def _safe_filename(key: str) -> str:
    """Make a session key safe to use as a filename.

    Keys come from device certificates, so they are not attacker-controlled in the
    normal case — but a key containing `../` would write outside the profile directory,
    and that is not a failure worth leaving available.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return cleaned[:128] or "unnamed"
