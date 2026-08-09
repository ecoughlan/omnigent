"""Helpers for seeding a managed Pi agent dir (``PI_CODING_AGENT_DIR``).

When the pi harness runs in gateway mode it relocates Pi's agent root to a
per-session temp directory (for ``models.json``). That hides the user's global
``~/.pi/agent/settings.json`` and installed package trees. This module copies
the global settings metadata and login state (``auth.json``) into the managed
dir and symlinks install directories so Pi's native loader still sees
extensions and ``pi install`` packages — without mutating the user's home
directory.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, cast

_logger = logging.getLogger(__name__)

_Settings: TypeAlias = dict[str, object]

# Default global Pi agent root (``PI_CODING_AGENT_DIR`` when unset).
DEFAULT_PI_AGENT_DIR = Path.home() / ".pi" / "agent"

# Install / checkout trees Pi places under the agent dir. Symlinked into the
# managed dir so ``packages`` entries in settings keep resolving.
_PI_AGENT_RESOURCE_DIRS: tuple[str, ...] = ("npm", "git")


def _read_settings_file(path: Path) -> _Settings:
    """Load a Pi ``settings.json`` file, returning ``{}`` when absent/invalid."""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning("Ignoring unreadable Pi settings at %s: %s", path, exc)
        return {}
    return cast(_Settings, raw) if isinstance(raw, dict) else {}


def _deep_merge_settings(base: Mapping[str, object], overlay: Mapping[str, object]) -> _Settings:
    """Merge *overlay* onto *base* using Pi's nested-object merge semantics."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_settings(existing, value)
        else:
            merged[key] = value
    return merged


def _symlink_agent_resource_dirs(
    managed_dir: Path,
    global_agent_dir: Path,
) -> None:
    """Symlink package install trees from *global_agent_dir* into *managed_dir*."""
    for name in _PI_AGENT_RESOURCE_DIRS:
        source = global_agent_dir / name
        if not source.is_dir():
            continue
        target = managed_dir / name
        if target.exists():
            continue
        try:
            target.symlink_to(source, target_is_directory=True)
        except OSError as exc:
            _logger.warning(
                "Could not symlink Pi agent resource %s -> %s: %s",
                target,
                source,
                exc,
            )


def _copy_global_auth_file(managed_dir: Path, global_agent_dir: Path) -> None:
    """Copy the user's global ``auth.json`` (CLI logins) into the managed dir.

    The managed dir shadows ``~/.pi/agent``, so without this the session's Pi
    starts with no login state: the model picker only sees ``models.json``
    providers, and models authenticated via Pi ``/login`` (Claude, GPT,
    DeepSeek, …) never appear as switchable. Copy (not symlink) so Pi's own
    in-session token refreshes never mutate the user's global file.
    """
    source = global_agent_dir / "auth.json"
    if not source.is_file():
        return
    try:
        data = source.read_bytes()
    except OSError as exc:
        _logger.warning("Could not read global Pi auth at %s: %s", source, exc)
        return
    target = managed_dir / "auth.json"
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except OSError as exc:
        _logger.warning("Could not copy Pi auth into managed dir at %s: %s", target, exc)


def prepare_managed_pi_agent_dir(
    managed_dir: Path,
    *,
    overlay: Mapping[str, object] | None = None,
    global_agent_dir: Path | None = None,
) -> None:
    """
    Seed a managed ``PI_CODING_AGENT_DIR`` with the user's global Pi settings.

    Copies (merges) ``settings.json`` from the user's global agent dir into
    *managed_dir*, applies *overlay* (e.g. Omnigent retry policy), copies the
    global ``auth.json`` (CLI logins) so the session's Pi can serve models
    authenticated via ``/login``, and symlinks install trees (``npm/``,
    ``git/``) so ``packages`` entries keep working. The user's
    ``~/.pi/agent`` is never modified.

    Project-scoped ``.pi/settings.json`` at the session cwd is still read by
    Pi at runtime and overrides these global settings per Pi's normal rules.

    :param managed_dir: Per-session agent root (already contains ``models.json``).
    :param overlay: Settings merged on top of the copied global file.
    :param global_agent_dir: Override for tests; defaults to
        :data:`DEFAULT_PI_AGENT_DIR`.
    """
    agent_root = global_agent_dir if global_agent_dir is not None else DEFAULT_PI_AGENT_DIR
    settings = _read_settings_file(agent_root / "settings.json")
    if overlay:
        settings = _deep_merge_settings(settings, overlay)

    managed_dir.mkdir(parents=True, exist_ok=True)
    settings_path = managed_dir / "settings.json"
    try:
        settings_path.write_text(
            json.dumps(settings, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        _logger.warning("Could not write managed Pi settings at %s: %s", settings_path, exc)
        return

    _copy_global_auth_file(managed_dir, agent_root)

    _symlink_agent_resource_dirs(managed_dir, agent_root)
