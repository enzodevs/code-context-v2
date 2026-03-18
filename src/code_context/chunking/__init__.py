"""Chunking module - AST-based code chunking with tree-sitter."""

from code_context.chunking.languages import LANGUAGE_CONFIG, detect_language
from code_context.chunking.models import ParsedChunk, compute_chunk_hash, compute_file_hash
from code_context.chunking.parser import CodeParser

__all__ = [
    "CodeParser",
    "LANGUAGE_CONFIG",
    "ParsedChunk",
    "compute_chunk_hash",
    "compute_file_hash",
    "detect_language",
]
