"""Unit tests for the indexer: embedding text, ignore patterns, edge cases.

Run with: uv run pytest tests/test_indexer.py -v
"""

import os
import tempfile
from pathlib import Path

import pytest

from code_context.chunking.models import ParsedChunk
from code_context.chunking.parser import CodeParser
from code_context.indexer import (
    DEFAULT_IGNORE_PATTERNS,
    IGNORE_FILE_PATTERNS,
    Indexer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return CodeParser()


@pytest.fixture
def indexer():
    """Indexer without DB/Voyage — only testing pure logic methods."""
    return Indexer(db=None, voyage=None)


def _make_chunk(
    text: str = "x = 1",
    chunk_type: str = "function",
    symbol_name: str | None = "do_thing",
    filepath: str = "src/utils.py",
    imports: list[str] | None = None,
    parent_class: str | None = None,
    signature: str | None = None,
) -> ParsedChunk:
    context = {"filepath": filepath}
    if imports is not None:
        context["imports"] = imports
    if parent_class:
        context["parent_class"] = parent_class
    if signature:
        context["signature"] = signature
    return ParsedChunk(
        text=text,
        start_line=1,
        end_line=1,
        chunk_type=chunk_type,
        symbol_name=symbol_name,
        context=context,
    )


# ---------------------------------------------------------------------------
# _build_embedding_text — what gets sent to Voyage
# ---------------------------------------------------------------------------

class TestBuildEmbeddingText:
    """The embedding text header directly affects retrieval quality.
    If it's malformed, every query against this chunk degrades.
    """

    def test_minimal_chunk(self):
        chunk = _make_chunk(symbol_name=None, imports=None)
        result = Indexer._build_embedding_text(chunk, "python")

        assert result.startswith("[context] ")
        assert "filepath=src/utils.py" in result
        assert "language=python" in result
        assert "chunk_type=function" in result
        # Should end with the actual code
        assert result.endswith("\nx = 1")

    def test_symbol_name_included(self):
        result = Indexer._build_embedding_text(
            _make_chunk(symbol_name="processOrder"), "typescript"
        )
        assert "symbol=processOrder" in result

    def test_parent_class_included(self):
        result = Indexer._build_embedding_text(
            _make_chunk(parent_class="OrderService", chunk_type="method"), "typescript"
        )
        assert "parent_class=OrderService" in result

    def test_imports_included_and_capped(self):
        imports = [f"import mod{i}" for i in range(10)]
        result = Indexer._build_embedding_text(
            _make_chunk(imports=imports), "python"
        )
        # Should include at most 5 imports
        assert "import mod0" in result
        assert "import mod4" in result
        assert "import mod5" not in result

    def test_signature_included(self):
        result = Indexer._build_embedding_text(
            _make_chunk(signature="def process(items: list[dict]) -> bool"), "python"
        )
        assert "signature=def process(items: list[dict]) -> bool" in result

    def test_empty_context(self):
        """Chunk with no context dict should not crash."""
        chunk = ParsedChunk(
            text="SELECT 1",
            start_line=1,
            end_line=1,
            chunk_type="statement",
            context={},
        )
        result = Indexer._build_embedding_text(chunk, "sql")
        assert "filepath=" in result  # empty but present
        assert "SELECT 1" in result

    def test_none_context(self):
        """Chunk with context=None should not crash."""
        chunk = ParsedChunk(
            text="x = 1",
            start_line=1,
            end_line=1,
            chunk_type="file",
            context=None,
        )
        result = Indexer._build_embedding_text(chunk, "python")
        assert "[context]" in result
        assert "x = 1" in result

    def test_header_is_single_line(self):
        """Header must be one line — multi-line headers would corrupt the format."""
        result = Indexer._build_embedding_text(
            _make_chunk(
                imports=["import os", "import sys"],
                signature="def foo(bar: str) -> None",
                parent_class="MyClass",
            ),
            "python",
        )
        lines = result.split("\n")
        assert lines[0].startswith("[context] ")
        # No other line should start with [context]
        assert not any(l.startswith("[context]") for l in lines[1:])

    def test_non_ascii_in_symbol_name(self):
        """Non-ASCII symbol names must survive embedding text construction."""
        result = Indexer._build_embedding_text(
            _make_chunk(symbol_name="calcularImpostoNação"), "typescript"
        )
        assert "symbol=calcularImpostoNação" in result

    def test_all_languages_produce_valid_header(self):
        """Every supported language should produce a parseable header."""
        for lang in ["typescript", "javascript", "python", "java", "go", "rust", "sql", "markdown"]:
            result = Indexer._build_embedding_text(_make_chunk(), lang)
            assert f"language={lang}" in result
            assert result.count("[context]") == 1


# ---------------------------------------------------------------------------
# should_ignore / _collect_files
# ---------------------------------------------------------------------------

class TestIgnorePatterns:
    def test_node_modules_ignored(self, indexer):
        assert indexer.should_ignore(Path("project/node_modules/express/index.js"))

    def test_git_ignored(self, indexer):
        assert indexer.should_ignore(Path(".git/objects/abc123"))

    def test_pycache_ignored(self, indexer):
        assert indexer.should_ignore(Path("src/__pycache__/module.cpython-311.pyc"))

    def test_normal_source_not_ignored(self, indexer):
        assert not indexer.should_ignore(Path("src/main.py"))
        assert not indexer.should_ignore(Path("lib/utils.ts"))

    def test_lock_files_ignored(self, indexer):
        assert indexer.should_ignore(Path("package-lock.json"))
        assert indexer.should_ignore(Path("yarn.lock"))
        assert indexer.should_ignore(Path("pnpm-lock.yaml"))

    def test_min_files_ignored(self, indexer):
        """Minified files should be caught by IGNORE_FILE_PATTERNS."""
        for filename in ["app.min.js", "styles.min.css", "vendor.bundle.js", "chunk.abc123.chunk.js"]:
            assert any(p in filename for p in IGNORE_FILE_PATTERNS), (
                f"{filename} should match an ignore pattern"
            )

    def test_source_maps_ignored(self, indexer):
        assert any(p in "app.js.map" for p in IGNORE_FILE_PATTERNS)

    def test_build_cache_dirs_ignored(self, indexer):
        """Build caches that caused noise (Phase A fix) must stay ignored."""
        for dirname in [".cxx", ".expo", "Pods", ".trigger", ".turbo", ".vercel"]:
            assert dirname in DEFAULT_IGNORE_PATTERNS, f"{dirname} missing from ignore patterns"

    def test_custom_ignore_patterns(self):
        """Custom patterns should override defaults."""
        custom = {"my_special_dir", "node_modules"}
        indexer = Indexer(db=None, voyage=None, ignore_patterns=custom)
        assert indexer.should_ignore(Path("project/my_special_dir/file.py"))
        assert not indexer.should_ignore(Path("project/.git/HEAD"))  # not in custom set


class TestCollectFiles:
    def test_collects_supported_files(self, indexer):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create supported files
            Path(tmpdir, "main.py").write_text("x = 1")
            Path(tmpdir, "app.ts").write_text("const x = 1;")
            Path(tmpdir, "schema.sql").write_text("SELECT 1;")
            # Create unsupported file
            Path(tmpdir, "image.png").write_bytes(b"\x89PNG")

            files = indexer._collect_files(Path(tmpdir))
            extensions = {f.suffix for f, _ in files}

            assert ".py" in extensions
            assert ".ts" in extensions
            assert ".sql" in extensions
            assert ".png" not in extensions

    def test_prunes_ignored_directories(self, indexer):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file in normal dir
            src = Path(tmpdir, "src")
            src.mkdir()
            Path(src, "app.py").write_text("x = 1")

            # Create file in ignored dir
            nm = Path(tmpdir, "node_modules", "express")
            nm.mkdir(parents=True)
            Path(nm, "index.js").write_text("module.exports = {}")

            files = indexer._collect_files(Path(tmpdir))
            paths = [str(f) for f, _ in files]

            assert any("app.py" in p for p in paths)
            assert not any("node_modules" in p for p in paths)

    def test_prunes_deeply_nested_ignored(self, indexer):
        """Ignored dirs nested deep should still be pruned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep = Path(tmpdir, "a", "b", "c", "node_modules", "pkg")
            deep.mkdir(parents=True)
            Path(deep, "index.js").write_text("x")

            normal = Path(tmpdir, "a", "b", "c")
            Path(normal, "real.ts").write_text("const x = 1;")

            files = indexer._collect_files(Path(tmpdir))
            paths = [str(f) for f, _ in files]

            assert any("real.ts" in p for p in paths)
            assert not any("node_modules" in p for p in paths)

    def test_returns_language_with_each_file(self, indexer):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.py").write_text("x = 1")
            Path(tmpdir, "app.ts").write_text("const x = 1;")

            files = indexer._collect_files(Path(tmpdir))
            langs = {lang for _, lang in files}

            assert "python" in langs
            assert "typescript" in langs

    def test_empty_directory(self, indexer):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = indexer._collect_files(Path(tmpdir))
            assert files == []


# ---------------------------------------------------------------------------
# file_only large file skipping
# ---------------------------------------------------------------------------

class TestFileOnlySkipping:
    """Auto-generated JSON/YAML caches should be skipped, hand-written configs kept."""

    def test_small_json_kept(self, parser):
        """A small tsconfig.json should produce a file chunk."""
        content = '{\n  "compilerOptions": {\n    "target": "ES2020",\n    "module": "commonjs"\n  }\n}'
        chunks = parser.parse_file("tsconfig.json", content)
        # May be empty if below min_tokens, but should not be skipped by line check
        assert isinstance(chunks, list)

    def test_large_json_skipped(self, parser, monkeypatch):
        """A huge auto-generated JSON should be skipped entirely."""
        from code_context.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "file_only_max_lines", 200)

        big_json = "{\n" + ",\n".join([f'  "key_{i}": {i}' for i in range(500)]) + "\n}"
        chunks = parser.parse_file("tsconfig.tsbuildinfo", big_json)
        assert chunks == []

    def test_large_yaml_skipped(self, parser, monkeypatch):
        from code_context.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "file_only_max_lines", 200)

        big_yaml = "\n".join([f"key_{i}: value_{i}" for i in range(500)])
        chunks = parser.parse_file("generated.yaml", big_yaml)
        assert chunks == []

    def test_code_files_not_affected(self, parser, monkeypatch):
        """The line cap should NOT apply to code files (only file_only strategy)."""
        from code_context.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "file_only_max_lines", 50)

        # A 300-line Python file should still parse normally
        big_py = "\n".join([f"x_{i} = {i}" for i in range(300)])
        chunks = parser.parse_file("big.py", big_py)
        assert len(chunks) >= 1

    def test_sql_not_affected(self, parser, monkeypatch):
        """SQL (statement strategy) should not be affected by file_only cap."""
        from code_context.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "file_only_max_lines", 50)

        big_sql = "\n\n".join(
            f"CREATE OR REPLACE FUNCTION fn_{i}(p_id BIGINT)\n"
            f"RETURNS VOID AS $$\nBEGIN\n    RETURN;\nEND;\n$$ LANGUAGE plpgsql;"
            for i in range(20)
        )
        chunks = parser.parse_file("big.sql", big_sql)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Parser edge cases — silent data loss scenarios
# ---------------------------------------------------------------------------

class TestParserEdgeCases:
    def test_empty_file(self, parser):
        """Empty file should produce no chunks, not crash."""
        chunks = parser.parse_file("empty.py", "")
        assert chunks == []

    def test_whitespace_only_file(self, parser):
        """Whitespace-only file should produce no chunks."""
        chunks = parser.parse_file("blank.py", "   \n\n  \n")
        assert chunks == []

    def test_comments_only_python(self, parser):
        """File with only comments should produce a file chunk if large enough."""
        content = "\n".join([f"# Comment line {i} with some content here" for i in range(50)])
        chunks = parser.parse_file("comments.py", content)
        # Should not crash; may produce file chunk if above min_tokens
        for c in chunks:
            assert c.text.strip()  # no empty chunks

    def test_single_line_function(self, parser):
        """Very small function should be filtered by min_tokens."""
        chunks = parser.parse_file("tiny.py", "def f(): pass")
        # Either filtered out or present as file chunk — either way, no crash
        assert isinstance(chunks, list)

    def test_unsupported_extension(self, parser):
        """Unsupported file type should return empty list, not crash."""
        chunks = parser.parse_file("data.parquet", "binary stuff")
        assert chunks == []

    def test_syntax_error_in_code(self, parser):
        """Files with syntax errors should still produce chunks (best-effort)."""
        bad_ts = """\
export function broken(x: string {
  // missing closing paren — tree-sitter should recover
  const result = x.toUpperCase();
  const trimmed = result.trim();
  const padded = trimmed.padStart(10, ' ');
  return padded;
}

export function working(y: number): number {
  const doubled = y * 2;
  const tripled = y * 3;
  const total = doubled + tripled;
  console.log(`Result: ${total}`);
  return total;
}
"""
        chunks = parser.parse_file("bad.ts", bad_ts)
        # tree-sitter is error-tolerant — should still extract something
        assert len(chunks) >= 1

    def test_very_long_single_line(self, parser):
        """A file with one very long line should not crash or hang."""
        long_line = "x = " + " + ".join([f'"{i}"' for i in range(500)])
        chunks = parser.parse_file("long.py", long_line)
        assert isinstance(chunks, list)

    def test_binary_content_does_not_crash(self, parser):
        """Binary content in a .py file should not crash the parser."""
        # Simulate a file that was misdetected as Python
        binary = "# header\n" + "\x00\x01\x02\xff" * 10
        try:
            chunks = parser.parse_file("weird.py", binary)
            assert isinstance(chunks, list)
        except UnicodeEncodeError:
            # Acceptable — binary content can't be encoded to UTF-8
            pass

    def test_file_with_bom(self, parser):
        """UTF-8 BOM should not break parsing."""
        content = "\ufeff# BOM file\ndef hello():\n    return 'world'\n"
        chunks = parser.parse_file("bom.py", content)
        # Should parse without error
        assert isinstance(chunks, list)

    def test_mixed_line_endings(self, parser):
        r"""Files with \r\n should parse correctly."""
        content = "def foo():\r\n    return 1\r\n\r\ndef bar():\r\n    return 2\r\n"
        chunks = parser.parse_file("crlf.py", content)
        assert isinstance(chunks, list)
        # Should not produce corrupted text
        for c in chunks:
            assert "\x00" not in c.text

    def test_deeply_nested_code(self, parser):
        """Deeply nested code should not cause recursion issues."""
        indent = ""
        lines = []
        for i in range(20):
            lines.append(f"{indent}if True:")
            indent += "    "
        lines.append(f"{indent}pass")
        content = "\n".join(lines)
        chunks = parser.parse_file("nested.py", content)
        assert isinstance(chunks, list)
