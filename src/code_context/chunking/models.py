"""Data models for parsed code chunks."""

from dataclasses import dataclass, field

import blake3
import tiktoken
from tree_sitter import Node

_encoder = tiktoken.get_encoding("cl100k_base")


@dataclass
class ParsedChunk:
    """A parsed code chunk with metadata."""

    text: str
    start_line: int
    end_line: int
    chunk_type: str  # file | declaration | function | class | method | section | statement
    symbol_name: str | None = None
    context: dict = field(default_factory=dict)
    _token_count: int = field(default=0, repr=False)
    _node: Node | None = field(default=None, repr=False, compare=False)

    @property
    def token_count(self) -> int:
        if self._token_count == 0:
            self._token_count = len(_encoder.encode(self.text))
        return self._token_count

    @token_count.setter
    def token_count(self, value: int) -> None:
        self._token_count = value


def compute_file_hash(content: str) -> str:
    """Compute BLAKE3 hash of file content."""
    return blake3.blake3(content.encode("utf-8")).hexdigest()


def compute_chunk_hash(text: str, filepath: str, start_line: int) -> str:
    """Compute hash for a chunk (for deduplication)."""
    data = f"{filepath}:{start_line}:{text}"
    return blake3.blake3(data.encode("utf-8")).hexdigest()[:32]
