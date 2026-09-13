from __future__ import annotations

import base64
import re
import unicodedata

ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
SPACE_SPLIT = re.compile(r"\b(?:f\s*o\s*r\s*g\s*e\s*t|i\s*g\s*n\s*o\s*r\s*e)\b", re.I)
PATTERNS = (
    r"ignore\s+(?:all\s+)?previous\s+instructions?",
    r"disregard\s+(?:the\s+)?policy",
    r"override\s+(?:the\s+)?controls?",
    r"forget\s+(?:all\s+)?(?:rules|instructions?)",
    r"pretend\s+(?:the\s+)?system\s+says",
    r"fake\s+(?:system\s+message|tool\s+output|authorization|approval\s+token)",
    r"role[ -]?play",
    r"approval\s+token.{0,30}(?:valid|approved|bypass)",
)
RISK = re.compile("|".join(f"(?:{item})" for item in PATTERNS), re.I | re.S)


def normalize_instruction_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", ZERO_WIDTH.sub("", value)).casefold()
    text = re.sub(r"[_/|:;,.!?'\"`~^*+-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return SPACE_SPLIT.sub(lambda match: re.sub(r"\s", "", match.group(0)), text)


def _decoded_candidates(value: str) -> list[str]:
    candidates = [value]
    for token in re.findall(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])", value):
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        candidates.append(decoded)
    return candidates


def detect_instruction_attack(value: str) -> bool:
    return any(
        RISK.search(normalize_instruction_text(candidate)) is not None for candidate in _decoded_candidates(value)
    )
