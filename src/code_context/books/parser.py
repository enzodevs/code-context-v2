"""EPUB parser for extracting chapters and text."""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

import tiktoken

# Suppress XML parsing warning - EPUBs often mix HTML/XML
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class BookChapter:
    """Represents a chapter or section from a book."""

    title: str
    content: str
    order: int


@dataclass
class BookMetadata:
    """Book metadata extracted from EPUB."""

    title: str
    author: str
    chapters: list[BookChapter]


@dataclass
class TextChunk:
    """A chunk of text ready for embedding."""

    chapter_title: str
    section: str | None
    text: str
    chunk_index: int
    token_count: int


class EpubParser:
    """Parse EPUB files and extract chapters with chunking."""

    def __init__(self, max_chunk_tokens: int = 500, overlap_tokens: int = 50):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def parse(self, filepath: Path) -> BookMetadata:
        """Parse an EPUB file and extract metadata and chapters."""
        book = epub.read_epub(str(filepath), options={"ignore_ncx": True})

        title = self._get_metadata(book, "title") or filepath.stem
        author = self._get_metadata(book, "creator") or "Unknown"

        chapters = []
        for idx, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
            html_content = item.get_content()
            soup = BeautifulSoup(html_content, "lxml")
            content = self._extract_text(html_content, soup=soup)
            if content.strip():
                chapter_title = self._extract_chapter_title(item, content, idx, soup=soup)
                chapters.append(BookChapter(title=chapter_title, content=content, order=idx))

        return BookMetadata(title=title, author=author, chapters=chapters)

    def chunk_book(self, metadata: BookMetadata) -> list[TextChunk]:
        """Split book chapters into chunks suitable for embedding."""
        chunks = []
        chunk_idx = 0

        for chapter in metadata.chapters:
            chapter_chunks = self._chunk_text(chapter.content, chapter.title)
            for text, section in chapter_chunks:
                token_count = len(self.tokenizer.encode(text))
                chunks.append(
                    TextChunk(
                        chapter_title=chapter.title,
                        section=section,
                        text=text,
                        chunk_index=chunk_idx,
                        token_count=token_count,
                    )
                )
                chunk_idx += 1

        return chunks

    def _get_metadata(self, book: epub.EpubBook, field: str) -> str | None:
        """Extract metadata field from EPUB."""
        try:
            values = book.get_metadata("DC", field)
            if values:
                return values[0][0]
        except Exception:
            pass
        return None

    def _extract_text(self, html_content: bytes, soup: BeautifulSoup | None = None) -> str:
        """Extract clean text from HTML content."""
        if soup is None:
            soup = BeautifulSoup(html_content, "lxml")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Clean up multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _extract_chapter_title(
        self,
        item: epub.EpubItem,
        _content: str,
        fallback_idx: int,
        soup: BeautifulSoup | None = None,
    ) -> str:
        """Extract chapter title from item or content."""
        # Try item title
        if hasattr(item, "title") and item.title:
            return item.title

        # Try first heading from content
        heading_source = soup if soup is not None else BeautifulSoup(item.get_content(), "lxml")
        for tag in ["h1", "h2", "h3"]:
            heading = heading_source.find(tag)
            if heading:
                title = heading.get_text(strip=True)
                if title and len(title) < 200:
                    return title

        # Fallback
        return f"Section {fallback_idx + 1}"

    def _chunk_text(self, text: str, chapter_title: str) -> list[tuple[str, str | None]]:
        """Split text into chunks with optional section detection."""
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0
        current_section = None

        for para in paragraphs:
            # Detect section headers (short lines that look like headers)
            if len(para) < 100 and para.isupper() or para.endswith(":"):
                current_section = para

            para_tokens = len(self.tokenizer.encode(para))

            # If single paragraph exceeds limit, split it
            if para_tokens > self.max_chunk_tokens:
                # Flush current chunk
                if current_chunk:
                    chunks.append(("\n\n".join(current_chunk), current_section))
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sentence_chunk = []
                sentence_tokens = 0

                for sentence in sentences:
                    s_tokens = len(self.tokenizer.encode(sentence))
                    if sentence_tokens + s_tokens > self.max_chunk_tokens and sentence_chunk:
                        chunks.append((" ".join(sentence_chunk), current_section))
                        sentence_chunk = []
                        sentence_tokens = 0
                    sentence_chunk.append(sentence)
                    sentence_tokens += s_tokens

                if sentence_chunk:
                    chunks.append((" ".join(sentence_chunk), current_section))
                continue

            # Normal case: accumulate paragraphs
            if current_tokens + para_tokens > self.max_chunk_tokens and current_chunk:
                chunks.append(("\n\n".join(current_chunk), current_section))
                # Keep last paragraph for overlap
                if self.overlap_tokens > 0 and current_chunk:
                    last = current_chunk[-1]
                    last_tokens = len(self.tokenizer.encode(last))
                    if last_tokens < self.overlap_tokens:
                        current_chunk = [last]
                        current_tokens = last_tokens
                    else:
                        current_chunk = []
                        current_tokens = 0
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_chunk:
            chunks.append(("\n\n".join(current_chunk), current_section))

        return chunks


def generate_book_id(title: str) -> str:
    """Generate a slug ID from book title."""
    # Common abbreviations
    abbreviations = {
        "designing data-intensive applications": "ddia",
        "the pragmatic programmer": "pragmatic-programmer",
        "ai engineering": "ai-engineering",
    }

    lower_title = title.lower()
    for pattern, abbrev in abbreviations.items():
        if pattern in lower_title:
            return abbrev

    # Generic slugify
    slug = re.sub(r"[^\w\s-]", "", lower_title)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")

    # Limit length
    return slug[:50]
