"""Section-aware chunking for synthetic clinical guideline documents.

Default behaviour uses a section-sentence strategy:
- preserve guideline section metadata;
- keep short sections as one chunk;
- split long sections on sentence boundaries;
- avoid splitting conditional rules from their following sentence;
- produce stable, deterministic chunk IDs for retrieval evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from typing import Dict, Iterable, List, Optional, Sequence

try:  # Supports both package imports and direct script execution.
    from .document_parser import CANONICAL_SECTIONS, parse_document
except ImportError:  # pragma: no cover
    from document_parser import CANONICAL_SECTIONS, parse_document


SECTION_SLUGS: Dict[str, str] = {
    "Background": "background",
    "Population": "population",
    "Intervention": "intervention",
    "Monitoring": "monitoring",
    "Outcomes": "outcomes",
    "Risks": "risks",
    "Evidence Summary": "evidence_summary",
}


@dataclass(frozen=True)
class ChunkMetadata:
    document_type: str
    contains_noise: bool = False
    contains_contradiction: bool = False
    contains_ambiguity: bool = False
    supports_abstention: bool = False
    contains_conditional_rule: bool = False


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    source_path: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        data = asdict(self)
        data["metadata"] = asdict(self.metadata)
        return data


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9≥≤\"“])")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def chunk_document(
    document: dict,
    *,
    strategy: str = "section_sentence",
    max_tokens: int = 180,
    min_tokens: int = 40,
    overlap_sentences: int = 1,
) -> List[dict]:
    """Chunk a parsed document.

    Args:
        document: Output of ``document_parser.parse_document``.
        strategy: ``section`` or ``section_sentence``.
        max_tokens: Maximum token budget for sentence-grouped chunks.
        min_tokens: Minimum preferred size before emitting non-final chunks.
        overlap_sentences: Number of sentences to carry into the next chunk.

    Returns:
        List of serialisable chunk dictionaries.
    """
    if strategy not in {"section", "section_sentence"}:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    metadata = infer_chunk_metadata(document)
    chunks: List[DocumentChunk] = []

    for section_name in CANONICAL_SECTIONS:
        section = document["sections"].get(section_name)
        if not section:
            continue

        section_text = normalise_chunk_text(section["text"])
        if not section_text:
            continue

        if strategy == "section" or count_tokens(section_text) <= max_tokens:
            chunk_texts = [section_text]
            offsets = [(section["char_start"], section["char_end"])]
        else:
            chunk_texts = split_section_into_sentence_chunks(
                section_text,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                overlap_sentences=overlap_sentences,
            )
            offsets = estimate_offsets(document["raw_text"], chunk_texts, section["char_start"])

        section_slug = SECTION_SLUGS[section_name]
        for idx, (chunk_text, (char_start, char_end)) in enumerate(zip(chunk_texts, offsets), start=1):
            chunk = DocumentChunk(
                chunk_id=f"{document['doc_id']}_{section_slug}_chunk_{idx:03d}",
                doc_id=document["doc_id"],
                title=document["title"],
                section=section_name,
                chunk_index=idx,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                token_count=count_tokens(chunk_text),
                source_path=document["source_path"],
                metadata=metadata,
            )
            chunks.append(chunk)

    return [chunk.to_dict() for chunk in chunks]

def chunk_by_section(document: dict, source_path: str):
    chunks = []

    for section_name, text in document["sections"].items():
        chunk = {
            "chunk_id": f"{document['doc_id']}_{section_name.lower()}_001",
            "doc_id": document["doc_id"],
            "section": section_name,
            "text": text,
            "source_path": source_path
        }
        chunks.append(chunk)

    return chunks

def chunk_file(
    path: str | Path,
    *,
    strategy: str = "section_sentence",
    max_tokens: int = 180,
    min_tokens: int = 40,
    overlap_sentences: int = 1,
) -> List[dict]:
    """Parse and chunk one document file."""
    document = parse_document(path)
    return chunk_document(
        document,
        strategy=strategy,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_sentences=overlap_sentences,
    )


def chunk_files(
    paths: Iterable[str | Path],
    *,
    strategy: str = "section_sentence",
    max_tokens: int = 180,
    min_tokens: int = 40,
    overlap_sentences: int = 1,
) -> List[dict]:
    """Chunk multiple files in deterministic order."""
    all_chunks: List[dict] = []
    for path in sorted(map(Path, paths)):
        all_chunks.extend(
            chunk_file(
                path,
                strategy=strategy,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                overlap_sentences=overlap_sentences,
            )
        )
    return all_chunks


def write_jsonl(records: Sequence[dict], output_path: str | Path) -> None:
    """Write records to JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



def split_section_into_sentence_chunks(
    text: str,
    *,
    max_tokens: int,
    min_tokens: int,
    overlap_sentences: int,
) -> List[str]:
    """Split section text into token-budgeted sentence chunks.

    Conditional labels such as ``Conditional monitoring rule:`` are attached to
    the next sentence to prevent orphaned rule headings.
    """
    sentences = sentence_split(text)
    sentences = attach_conditional_labels(sentences)

    chunks: List[str] = []
    current: List[str] = []

    for sentence in sentences:
        candidate = current + [sentence]
        candidate_tokens = count_tokens(" ".join(candidate))

        if current and candidate_tokens > max_tokens and count_tokens(" ".join(current)) >= min_tokens:
            chunks.append(" ".join(current).strip())
            if overlap_sentences > 0:
                current = current[-overlap_sentences:] + [sentence]
            else:
                current = [sentence]
        else:
            current = candidate

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def sentence_split(text: str) -> List[str]:
    """Sentence split with lightweight handling of line-based bullets."""
    cleaned = re.sub(r"\n+", " ", text.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return []

    parts = _SENTENCE_BOUNDARY_RE.split(cleaned)
    return [part.strip() for part in parts if part.strip()]


def attach_conditional_labels(sentences: List[str]) -> List[str]:
    """Attach conditional rule labels to their following sentence."""
    merged: List[str] = []
    skip_next = False
    for index, sentence in enumerate(sentences):
        if skip_next:
            skip_next = False
            continue
        if sentence.lower().startswith("conditional") and sentence.endswith(":") and index + 1 < len(sentences):
            merged.append(f"{sentence} {sentences[index + 1]}")
            skip_next = True
        else:
            merged.append(sentence)
    return merged


def infer_chunk_metadata(document: dict) -> ChunkMetadata:
    """Infer coarse document-level metadata flags for slice evaluation."""
    text = document["raw_text"].lower()
    title = document["title"].lower()

    return ChunkMetadata(
        document_type=infer_document_type(title, text),
        contains_noise=any(
            phrase in text
            for phrase in (
                "not directly relevant",
                "unrelated",
                "duplicated",
                "seasonal variation",
                "does not apply",
            )
        ),
        contains_contradiction=any(
            phrase in text
            for phrase in (
                "conflicting evidence",
                "evidence is inconsistent",
                "evidence is mixed",
                "evidence is inconclusive",
                "no statistically significant improvement",
                "differs from other studies",
                "although other reports",
            )
        ),
        contains_ambiguity=any(
            phrase in text
            for phrase in (
                "not standardised",
                "depending on clinical judgement",
                "not consistently defined",
                "inconsistent",
                "uncertain thresholds",
                "clinically meaningful",
            )
        ),
        supports_abstention=any(
            phrase in text
            for phrase in (
                "no reliable outcome data exist",
                "no standard monitoring recommendations are available",
                "not formally evaluated",
                "limited evidence",
                "potential risks are unknown",
            )
        ),
        contains_conditional_rule="conditional" in text or re.search(r"\bif\b", text) is not None,
    )


def infer_document_type(title: str, text: str) -> str:
    """Infer a stable document type label from title/content."""
    if "limited evidence" in title or "not formally evaluated" in text:
        return "insufficient_evidence"
    if "uncertain" in title or "not standardised" in text:
        return "ambiguous_evidence"
    if "conflicting" in title or "inconsistent" in text:
        return "conflicting_evidence"
    if "discontinuation" in title or "reinitiation" in title:
        return "discontinuation_protocol"
    if "adaptive" in title or "event-driven" in text:
        return "event_driven_monitoring"
    if "preventive" in title:
        return "preventive_guideline"
    if "dose adjustment" in title:
        return "dose_adjustment"
    if "acute" in title or "short-term" in title or "single dose" in text:
        return "acute_intervention"
    if "risk-stratified" in title or "risk factors" in title:
        return "risk_stratified_management"
    return "chronic_management"


def validate_chunk(chunk: dict) -> List[str]:
    """Return quality warnings for one chunk."""
    warnings: List[str] = []
    required_keys = {"chunk_id", "doc_id", "title", "section", "text", "token_count"}
    missing = required_keys - set(chunk)
    if missing:
        warnings.append(f"Missing keys: {', '.join(sorted(missing))}")

    if not chunk.get("text", "").strip():
        warnings.append("Chunk text is empty.")
    if chunk.get("token_count", 0) < 10:
        warnings.append("Chunk is very short; verify it is useful for retrieval.")
    if chunk.get("section") not in CANONICAL_SECTIONS:
        warnings.append(f"Unknown section: {chunk.get('section')}")
    if chunk.get("char_start", -1) < 0 or chunk.get("char_end", -1) < chunk.get("char_start", 0):
        warnings.append("Invalid character offsets.")

    text = chunk.get("text", "")
    if re.search(r"Conditional[^.]*:$", text, flags=re.IGNORECASE):
        warnings.append("Conditional rule label appears orphaned at chunk end.")
    return warnings


def validate_chunks(chunks: Sequence[dict]) -> Dict[str, List[str]]:
    """Validate chunks and return warnings keyed by chunk ID."""
    report: Dict[str, List[str]] = {}
    seen_ids = set()
    for chunk in chunks:
        warnings = validate_chunk(chunk)
        chunk_id = chunk.get("chunk_id", "<missing_chunk_id>")
        if chunk_id in seen_ids:
            warnings.append("Duplicate chunk_id.")
        seen_ids.add(chunk_id)
        if warnings:
            report[chunk_id] = warnings
    return report


def count_tokens(text: str) -> int:
    """Approximate token count for chunk sizing without model dependencies."""
    return len(_TOKEN_RE.findall(text))


def normalise_chunk_text(text: str) -> str:
    """Normalise whitespace while preserving semantic punctuation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def estimate_offsets(raw_text: str, chunk_texts: Sequence[str], section_start: int) -> List[tuple[int, int]]:
    """Estimate chunk offsets by finding chunk starts in the raw text.

    The text is normalised for chunking, so offsets are best-effort. They are
    sufficient for traceability and can be tightened later if needed.
    """
    offsets: List[tuple[int, int]] = []
    cursor = section_start
    for chunk_text in chunk_texts:
        needle = chunk_text[: min(80, len(chunk_text))].strip()
        found_at = raw_text.find(needle, cursor)
        if found_at == -1:
            found_at = cursor
        char_end = min(found_at + len(chunk_text), len(raw_text))
        offsets.append((found_at, char_end))
        cursor = char_end
    return offsets


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    import argparse

    parser = argparse.ArgumentParser(description="Build section-aware chunks from synthetic documents.")
    parser.add_argument("input", nargs="+", help="Input .txt files")
    parser.add_argument("--output", help="Optional JSONL output path")
    parser.add_argument("--strategy", default="section_sentence", choices=["section", "section_sentence"])
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--min-tokens", type=int, default=40)
    parser.add_argument("--overlap-sentences", type=int, default=1)
    args = parser.parse_args()

    chunks = chunk_files(
        args.input,
        strategy=args.strategy,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        overlap_sentences=args.overlap_sentences,
    )
    if args.output:
        write_jsonl(chunks, args.output)
    else:
        for chunk in chunks:
            print(json.dumps(chunk, ensure_ascii=False))
