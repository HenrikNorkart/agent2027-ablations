"""Helpers for environment key aliasing and validation."""

from __future__ import annotations

import os


def apply_env_aliases(alias_map: dict[str, str]) -> None:
    """Apply backward-compatible env aliases.

    If source key exists and target key is missing, target gets source value.
    """
    for source_key, target_key in alias_map.items():
        if os.getenv(target_key) is None and os.getenv(source_key) is not None:
            os.environ[target_key] = os.getenv(source_key, "")


def apply_optional_defaults(defaults: dict[str, str]) -> None:
    """Populate optional defaults for missing keys."""
    for key, default in defaults.items():
        if os.getenv(key) is None:
            os.environ[key] = default


def validate_required_env_keys(required_keys: list[str]) -> None:
    """Raise an explicit error if any required env keys are missing or blank."""
    missing = [k for k in required_keys if os.getenv(k) in (None, "")]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise RuntimeError(
            "Missing required environment variables: "
            f"{missing_str}. Populate these in .env before startup."
        )


def env_as_bool(key: str, default: bool) -> bool:
    """Parse common boolean environment formats."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}
