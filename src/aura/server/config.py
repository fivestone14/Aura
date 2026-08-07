"""Configuration and credential resolution.

Design goal: bringing Aura up on a new machine should be one step — provide a key —
with everything else defaulted sensibly.

Credentials are resolved in a deliberate order and are **never** read from the repo.
The keychain is preferred over environment variables because a `.env` file is a
project file, and project files get indexed by development tooling
(docs/DESIGN.md §10).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Final

KEYCHAIN_SERVICE: Final = "aura"
"""Keychain service name. Store a key with:

    security add-generic-password -s aura -a anthropic -w

Note the omitted value — `security` will prompt. Passing `-w <secret>` on the command
line writes the secret to shell history permanently.
"""

ENV_VAR: Final = "ANTHROPIC_API_KEY"


class CredentialError(RuntimeError):
    """No usable credential was found. The message explains how to fix it."""


def resolve_api_key(account: str = "anthropic") -> str:
    """Find an API key, preferring the most secure source available.

    Order: macOS keychain, then environment. Nothing else — notably not a file in the
    working directory, which is how keys end up committed.
    """
    key = _from_keychain(account)
    if key:
        return key

    key = os.environ.get(ENV_VAR, "").strip()
    if key:
        return key

    raise CredentialError(
        "No API key found.\n\n"
        "  Recommended (macOS keychain, prompts for the value):\n"
        f"    security add-generic-password -s {KEYCHAIN_SERVICE} -a {account} -w\n\n"
        "  Or, for a shell session only:\n"
        f"    export {ENV_VAR}=...\n\n"
        "Do not put the key in a .env file or commit it."
    )


def _from_keychain(account: str) -> str | None:
    """Read from the macOS keychain, or None anywhere else."""
    security = shutil.which("security")
    if security is None:
        return None
    try:
        result = subprocess.run(
            [security, "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", account, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


# Defaults live at module scope rather than only as dataclass field defaults.
#
# With `slots=True` a dataclass has no class-level attributes — the names are slot
# descriptors instead. Reading `cls.model` inside a classmethod therefore returns the
# descriptor object, not "claude-opus-5", and does so *silently*: the Settings object
# constructs fine and the wrong value only surfaces when it reaches an API call.
DEFAULT_MODEL: Final = "claude-opus-5"
DEFAULT_COUNCIL_DEADLINE: Final = 2.0
DEFAULT_BASELINE_WINDOW: Final = 50
DEFAULT_MAX_REPLY_TOKENS: Final = 512


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything tunable, with defaults that work unconfigured.

    Values come from the environment so the same build runs on a laptop and a host
    without edits — no machine-specific paths, no committed config
    (docs/DESIGN.md §7).
    """

    model: str = DEFAULT_MODEL
    council_deadline_seconds: float = DEFAULT_COUNCIL_DEADLINE
    baseline_window: int = DEFAULT_BASELINE_WINDOW
    max_reply_tokens: int = DEFAULT_MAX_REPLY_TOKENS

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from `AURA_*` environment variables, falling back to defaults."""
        return cls(
            model=os.environ.get("AURA_MODEL", DEFAULT_MODEL),
            council_deadline_seconds=_env_float(
                "AURA_COUNCIL_DEADLINE", DEFAULT_COUNCIL_DEADLINE
            ),
            baseline_window=_env_int("AURA_BASELINE_WINDOW", DEFAULT_BASELINE_WINDOW),
            max_reply_tokens=_env_int("AURA_MAX_REPLY_TOKENS", DEFAULT_MAX_REPLY_TOKENS),
        )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
