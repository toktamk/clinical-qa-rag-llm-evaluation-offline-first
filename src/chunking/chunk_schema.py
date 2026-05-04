from pydantic import BaseModel, Field
from typing import Optional


class ChunkMetadata(BaseModel):
    document_type: str
    contains_noise: bool = False
    contains_contradiction: bool = False
    contains_ambiguity: bool = False
    supports_abstention: bool = False
    contains_conditional_rule: bool = False


class DocumentChunk(BaseModel):
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