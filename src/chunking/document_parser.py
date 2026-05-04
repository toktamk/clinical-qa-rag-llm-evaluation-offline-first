"""Document parsing utilities for synthetic clinical guideline documents.

The parser is intentionally conservative and dependency-free. It supports the
current synthetic corpus where documents may contain:
- an explicit ``Doc ID: doc_001`` line, or only a filename such as ``doc_001.txt``;
- a ``Title: ...`` line, Markdown headings, or both;
- numbered sections such as ``1. Background``;
- Markdown headings such as ``### **1. Background**``.

The returned dictionary is designed to be consumed by ``section_chunker.py`` and
later QA/evaluation-data generation modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple


CANONICAL_SECTIONS: Tuple[str, ...] = (
    "Background",
    "Population",
    "Intervention",
    "Monitoring",
    "Outcomes",
    "Risks",
    "Evidence Summary",
)

SECTION_ALIASES: Dict[str, str] = {
    "risk": "Risks",
    "risks": "Risks",
    "limitations": "Risks",
    "risks / limitations": "Risks",
    "evidence": "Evidence Summary",
    "evidence summary": "Evidence Summary",
    "intervention / exposure": "Intervention",
    "intervention": "Intervention",
    "exposure": "Intervention",
    "background": "Background",
    "population": "Population",
    "monitoring": "Monitoring",
    "outcome": "Outcomes",
    "outcomes": "Outcomes",
}


@dataclass(frozen=True)
class ParsedSection:
    """A parsed section with character offsets in the original document."""

    name: str
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ParsedDocument:
    """Parsed representation of a synthetic guideline document."""

    doc_id: str
    title: str
    source_path: str
    sections: Dict[str, ParsedSection]
    raw_text: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a serialisable dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_path": self.source_path,
            "sections": {
                name: {
                    "name": section.name,
                    "text": section.text,
                    "char_start": section.char_start,
                    "char_end": section.char_end,
                }
                for name, section in self.sections.items()
            },
            "raw_text": self.raw_text,
            "warnings": list(self.warnings),
        }


_DOC_ID_RE = re.compile(r"^\s*(?:Doc\s*ID|Document\s*ID)\s*:\s*([A-Za-z0-9_\-]+)\s*$", re.IGNORECASE | re.MULTILINE)
_TITLE_RE = re.compile(r"^\s*(?:#+\s*)?(?:\*\*)?Title\s*:\s*(.*?)(?:\*\*)?\s*$", re.IGNORECASE | re.MULTILINE)

# Matches lines such as:
# 1. Background
# ### **1. Background**
# ## 7. Evidence Summary
_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(\d+)\.\s*([A-Za-z][A-Za-z\s/\-]+?)\s*(?:\*\*)?\s*$",
    re.MULTILINE,
)

_MARKDOWN_TITLE_RE = re.compile(r"^\s*#{1,3}\s*(?:\*\*)?(.+?)(?:\*\*)?\s*$", re.MULTILINE)


def normalise_text(text: str) -> str:
    """Normalise line endings and remove invisible formatting artefacts."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    # Keep paragraph boundaries but remove excessive trailing whitespace.
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def infer_doc_id(path: str | Path, text: str) -> str:
    """Infer document ID from explicit text metadata or filename."""
    match = _DOC_ID_RE.search(text)
    if match:
        return match.group(1).strip()

    stem = Path(path).stem
    filename_match = re.search(r"(doc[_\-]?\d+)", stem, flags=re.IGNORECASE)
    if filename_match:
        return filename_match.group(1).lower().replace("-", "_")
    return stem


def extract_title(path: str | Path, text: str, doc_id: str) -> Tuple[str, Optional[str]]:
    """Extract title and return optional warning."""
    match = _TITLE_RE.search(text)
    if match:
        return _clean_inline_markup(match.group(1)), None

    # Fallback: use first Markdown heading that is not a section heading.
    for heading in _MARKDOWN_TITLE_RE.finditer(text):
        value = _clean_inline_markup(heading.group(1))
        if not re.match(r"^\d+\.\s+", value):
            if not value.lower().startswith("doc id"):
                return value, None

    return doc_id.replace("_", " ").title(), "Missing explicit title; inferred from document ID."


def parse_document(path: str | Path) -> dict:
    """Parse a synthetic guideline document into metadata and sections.

    Args:
        path: Path to a ``.txt`` synthetic guideline document.

    Returns:
        A dictionary with keys: ``doc_id``, ``title``, ``source_path``,
        ``sections``, ``raw_text`` and ``warnings``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no recognisable guideline sections are found.
    """
    path = Path(path)
    raw_text = normalise_text(path.read_text(encoding="utf-8"))
    warnings: List[str] = []

    doc_id = infer_doc_id(path, raw_text)
    title, title_warning = extract_title(path, raw_text, doc_id)
    if title_warning:
        warnings.append(title_warning)

    sections = _extract_sections(raw_text)
    if not sections:
        raise ValueError(f"No recognised numbered sections found in {path}")

    missing_sections = [section for section in CANONICAL_SECTIONS if section not in sections]
    if missing_sections:
        warnings.append(f"Missing canonical sections: {', '.join(missing_sections)}")

    parsed = ParsedDocument(
        doc_id=doc_id,
        title=title,
        source_path=str(path),
        sections=sections,
        raw_text=raw_text,
        warnings=warnings,
    )
    return parsed.to_dict()


def parse_documents(paths: Iterable[str | Path]) -> List[dict]:
    """Parse multiple documents in deterministic path order."""
    return [parse_document(path) for path in sorted(map(Path, paths))]


def _extract_sections(text: str) -> Dict[str, ParsedSection]:
    matches = list(_SECTION_HEADING_RE.finditer(text))
    sections: Dict[str, ParsedSection] = {}

    for index, match in enumerate(matches):
        raw_name = _clean_inline_markup(match.group(2))
        canonical_name = _canonicalise_section_name(raw_name)
        if canonical_name is None:
            continue

        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[content_start:content_end].strip()

        if not section_text:
            continue

        # If duplicate section headings appear, keep the first name and append
        # later content with an explicit separator rather than silently dropping it.
        if canonical_name in sections:
            previous = sections[canonical_name]
            merged_text = previous.text.rstrip() + "\n\n" + section_text
            sections[canonical_name] = ParsedSection(
                name=canonical_name,
                text=merged_text,
                char_start=previous.char_start,
                char_end=content_end,
            )
        else:
            sections[canonical_name] = ParsedSection(
                name=canonical_name,
                text=section_text,
                char_start=content_start,
                char_end=content_end,
            )

    return sections


def _canonicalise_section_name(raw_name: str) -> Optional[str]:
    key = re.sub(r"\s+", " ", raw_name.strip().lower())
    return SECTION_ALIASES.get(key)


def _clean_inline_markup(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^#+\s*", "", value)
    value = value.replace("**", "")
    value = value.replace("📄", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


if __name__ == "__main__":  # pragma: no cover - convenience debugging entrypoint
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse synthetic clinical guideline documents.")
    parser.add_argument("paths", nargs="+", help="Document paths to parse")
    args = parser.parse_args()

    parsed_docs = parse_documents(args.paths)
    print(json.dumps(parsed_docs, indent=2, ensure_ascii=False))
