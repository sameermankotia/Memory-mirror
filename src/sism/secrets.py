"""Credential resolution.

Order of precedence:
  1. Explicit ``OPENROUTER_API_KEY`` in the environment.
  2. ``OPENROUTER_API_KEY`` in a local ``.env``.
  3. A 1Password secret reference (``SISM_OP_SECRET_REF``) read via the
     ``op`` CLI. This shells out to ``op read``, which requires an
     unlocked 1Password session (``eval "$(op signin)"``).

Nothing here ever logs or persists the key.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_OP_REF = "op://Private/OpenRouter/credential"


class MissingCredential(RuntimeError):
    pass


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


def _from_1password(ref: str) -> str | None:
    if not shutil.which("op"):
        return None
    try:
        out = subprocess.run(
            ["op", "read", ref],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def get_openrouter_key(*, required: bool = True) -> str | None:
    """Resolve the OpenRouter API key, or raise if it cannot be found."""
    _load_dotenv(Path.cwd() / ".env")

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    ref = os.environ.get("SISM_OP_SECRET_REF", DEFAULT_OP_REF).strip()
    if ref:
        key = _from_1password(ref) or ""
        if key:
            os.environ["OPENROUTER_API_KEY"] = key  # cache for the process only
            return key

    if not required:
        return None

    raise MissingCredential(
        "No OpenRouter API key found.\n"
        "  Fix it in any one of these ways:\n"
        "    export OPENROUTER_API_KEY=sk-or-...\n"
        "    cp .env.example .env   # then fill in the key\n"
        f'    eval "$(op signin)" && export SISM_OP_SECRET_REF="{ref}"\n'
        "  See README.md > Credentials."
    )


def redact(key: str | None) -> str:
    if not key:
        return "<none>"
    return f"{key[:7]}...{key[-4:]}" if len(key) > 14 else "<set>"
