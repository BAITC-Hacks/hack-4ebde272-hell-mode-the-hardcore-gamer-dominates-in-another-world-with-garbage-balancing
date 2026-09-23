"""Read local optional-AI settings without executing or exporting dotenv content.

Only OPENAI_API_KEY and OPENAI_MODEL are accepted. Nonempty process values take
precedence over the repository's .env file; blank values fall back to that file.
The file is read on every call, so a native Streamlit rerun sees a newly saved key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re


DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_MODEL = "gpt-4o-mini"
_ALLOWED_KEYS = frozenset({"OPENAI_API_KEY", "OPENAI_MODEL"})


@dataclass(frozen=True)
class AISettings:
    """Optional model configuration; representations never include the API key."""

    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL


def _literal_value(raw: str) -> str | None:
    """Parse one literal value with comments and optional single/double quotes.

    Inside quotes, backslash only escapes the matching quote or another backslash.
    Other backslashes, dollar signs and command-like text remain literal. A malformed
    quoted assignment is ignored; its contents never appear in logs or errors.
    """
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return ""
    if raw[0] not in {"'", '"'}:
        return re.split(r"\s+#", raw, maxsplit=1)[0].strip()
    quote, characters, index = raw[0], [], 1
    while index < len(raw):
        character = raw[index]
        if character == quote:
            remaining = raw[index + 1:].strip()
            return "".join(characters).strip() if not remaining or remaining.startswith("#") else None
        if character == "\\" and index + 1 < len(raw) and raw[index + 1] in {quote, "\\"}:
            index += 1
            character = raw[index]
        characters.append(character)
        index += 1
    return None


def _file_settings(path: Path) -> dict[str, str]:
    """Read only recognized assignments, tolerating absent or malformed files."""
    try:
        content = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return {}
    result = {}
    for line in content.splitlines():
        assignment = line.strip()
        if assignment.startswith("export ") or assignment.startswith("export\t"):
            assignment = assignment[6:].lstrip()
        key, separator, raw = assignment.partition("=")
        key = key.strip()
        if not separator or key not in _ALLOWED_KEYS:
            continue
        value = _literal_value(raw)
        if value is not None:
            result[key] = value
    return result


def get_ai_settings(env_path: str | Path | None = None) -> AISettings:
    """Load settings without mutating os.environ, interpolation, or key logging.

    Unknown keys and malformed assignments are ignored. Missing/unreadable files are
    equivalent to an empty file. A blank API key keeps AI optional; a blank model uses
    the documented default. ``env_path`` is useful for isolated tests and explicit
    local configuration locations.
    """
    values = _file_settings(Path(env_path) if env_path is not None else DEFAULT_ENV_PATH)

    def configured(name: str) -> str:
        return os.environ.get(name, "").strip() or values.get(name, "").strip()

    return AISettings(api_key=configured("OPENAI_API_KEY"),
                      model=configured("OPENAI_MODEL") or DEFAULT_MODEL)
