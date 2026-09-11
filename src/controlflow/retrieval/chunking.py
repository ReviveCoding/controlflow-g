from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    strategy: str
    ordinal: int
    parent_id: str | None = None


def fixed_chunks(document_id: str, text: str, size: int, overlap: int = 32) -> list[Chunk]:
    tokens = text.split()
    stride = max(1, size - overlap)
    return [
        Chunk(
            f"{document_id}:fixed-{size}:{index}",
            " ".join(tokens[start : start + size]),
            f"fixed_{size}",
            index,
        )
        for index, start in enumerate(range(0, len(tokens), stride))
        if tokens[start : start + size]
    ]


def section_chunks(document_id: str, text: str) -> list[Chunk]:
    sections = [section.strip() for section in re.split(r"(?m)^(?=[A-Z][A-Z0-9 .-]{3,}$)", text) if section.strip()]
    return [
        Chunk(f"{document_id}:section:{index}", section, "section_aware", index)
        for index, section in enumerate(sections)
    ]


def control_chunks(document_id: str, text: str) -> list[Chunk]:
    sections = [section.strip() for section in re.split(r"(?=\b[A-Z]{2}-\d+(?:\.\d+)?\b)", text) if section.strip()]
    return [
        Chunk(f"{document_id}:control:{index}", section, "control_aware", index)
        for index, section in enumerate(sections)
    ]


def hierarchical_chunks(document_id: str, text: str) -> list[Chunk]:
    parents = section_chunks(document_id, text)
    chunks: list[Chunk] = []
    for parent in parents:
        chunks.append(parent)
        chunks.extend(
            Chunk(child.chunk_id, child.text, "hierarchical", child.ordinal, parent.chunk_id)
            for child in fixed_chunks(parent.chunk_id, parent.text, 256)
        )
    return chunks
