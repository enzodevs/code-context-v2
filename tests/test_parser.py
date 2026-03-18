"""Unit tests for the chunking parser.

Covers: UTF-8 correctness, SQL parsing, language detection, module re-exports.

Run with: uv run pytest tests/test_parser.py -v
"""

import pytest

from code_context.chunking.parser import CodeParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return CodeParser()


# ---------------------------------------------------------------------------
# UTF-8 / non-ASCII byte-offset correctness
# ---------------------------------------------------------------------------

# Portuguese comments create byte drift (each ã/ç/é = 2 bytes in UTF-8).
# The bug: tree-sitter returns byte offsets, but code sliced a Python str
# (char offsets) — producing garbage after the first multi-byte char.

SQL_WITH_PORTUGUESE = """\
-- Configuração do schema platform para funções DDL
SET search_path TO platform;

/*
================================================================================
Descrição: Funções utilitárias para criação e validação de tabelas
================================================================================
*/

CREATE OR REPLACE FUNCTION ddl_create_table(p_schema TEXT, p_tabela TEXT)
RETURNS VOID AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = p_schema AND table_name = p_tabela
    ) THEN
        RETURN;
    END IF;
    EXECUTE FORMAT('CREATE TABLE IF NOT EXISTS %I.%I (id BIGSERIAL PRIMARY KEY)', p_schema, p_tabela);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION ddl_rename_table(p_schema TEXT, p_old TEXT, p_new TEXT)
RETURNS VOID AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = p_schema AND table_name = p_old
    ) THEN
        RETURN;
    END IF;
    EXECUTE FORMAT('ALTER TABLE %I.%I RENAME TO %I', p_schema, p_old, p_new);
END;
$$ LANGUAGE plpgsql;
"""

PYTHON_WITH_JAPANESE = """\
# このモジュールは認証処理を担当します
# ユーザーの資格情報を検証し、セッショントークンを発行します

def authenticate(username: str, password: str) -> bool:
    \"\"\"Authenticate user credentials.\"\"\"
    return username == "admin" and password == "secret"

def create_session(user_id: int) -> str:
    \"\"\"Create a new session token.\"\"\"
    import secrets
    return secrets.token_hex(32)
"""

TS_WITH_UNICODE_IDENTIFIERS = """\
// Módulo de configuração — importações necessárias
import { Config } from "./config";

export function autenticarUsuário(credenciais: { email: string; senha: string }): boolean {
  if (!credenciais.email) {
    throw new Error("Email obrigatório");
  }
  return true;
}

export function calcularImpostoNação(país: string, valor: number): number {
  const taxas: Record<string, number> = {
    "Brasil": 0.27,
    "México": 0.16,
    "España": 0.21,
  };
  return valor * (taxas[país] ?? 0);
}
"""


class TestUTF8ByteOffsets:
    """Regression tests for the UTF-8 byte-offset fix.

    Before the fix, any file with non-ASCII characters produced corrupted
    chunk text because tree-sitter byte offsets were used to slice a Python
    str (which uses char offsets). The corruption grew with each multi-byte
    character encountered before the extraction point.
    """

    def test_sql_function_names_after_portuguese_comments(self, parser):
        """SQL functions after Portuguese comments must have correct names."""
        chunks = parser.parse_file("test.sql", SQL_WITH_PORTUGUESE)
        stmt_names = [c.symbol_name for c in chunks if c.chunk_type == "statement"]

        assert "ddl_create_table" in stmt_names
        assert "ddl_rename_table" in stmt_names

    def test_sql_text_not_corrupted(self, parser):
        """Chunk text must start with CREATE, not shifted garbage."""
        chunks = parser.parse_file("test.sql", SQL_WITH_PORTUGUESE)

        for c in chunks:
            if c.chunk_type == "statement" and c.symbol_name:
                assert c.text.strip().startswith("CREATE"), (
                    f"Chunk {c.symbol_name!r} text corrupted: {c.text[:60]!r}"
                )

    def test_python_functions_after_japanese_comments(self, parser):
        """Python functions after CJK comments must parse correctly."""
        chunks = parser.parse_file("test.py", PYTHON_WITH_JAPANESE)
        func_chunks = [c for c in chunks if c.chunk_type in ("function", "declaration")]
        names = [c.symbol_name for c in func_chunks]

        assert "authenticate" in names
        assert "create_session" in names

    def test_python_last_function_text_integrity(self, parser):
        """The second function (after max byte drift) must have correct text."""
        chunks = parser.parse_file("test.py", PYTHON_WITH_JAPANESE)
        session_chunk = next(
            (c for c in chunks if c.symbol_name == "create_session"), None
        )
        assert session_chunk is not None

        assert "def create_session" in session_chunk.text
        assert "secrets.token_hex" in session_chunk.text

    def test_ts_unicode_identifiers_preserved(self, parser):
        """TS functions with unicode in params/body must preserve them."""
        chunks = parser.parse_file("test.ts", TS_WITH_UNICODE_IDENTIFIERS)
        fn_chunks = [c for c in chunks if c.chunk_type in ("function", "declaration")]

        tax_chunk = next(
            (c for c in fn_chunks if c.symbol_name == "calcularImpostoNação"), None
        )
        assert tax_chunk is not None, (
            f"Expected calcularImpostoNação, found: {[c.symbol_name for c in fn_chunks]}"
        )
        assert "país" in tax_chunk.text
        assert "Brasil" in tax_chunk.text

    def test_ts_imports_collected_after_non_ascii(self, parser):
        """Import collection must work when non-ASCII appears before imports."""
        chunks = parser.parse_file("test.ts", TS_WITH_UNICODE_IDENTIFIERS)

        for c in chunks:
            imports = c.context.get("imports", [])
            if imports:
                assert any("Config" in imp for imp in imports), (
                    f"Import not captured correctly: {imports}"
                )
                break

    def test_no_replacement_chars_in_chunks(self, parser):
        """No chunk should contain U+FFFD replacement characters."""
        for suffix, content in [
            ("test.sql", SQL_WITH_PORTUGUESE),
            ("test.py", PYTHON_WITH_JAPANESE),
            ("test.ts", TS_WITH_UNICODE_IDENTIFIERS),
        ]:
            chunks = parser.parse_file(suffix, content)
            for c in chunks:
                assert "\ufffd" not in c.text, (
                    f"Replacement char in {suffix} chunk {c.symbol_name!r}"
                )
                assert "\x00" not in c.text, (
                    f"Null byte in {suffix} chunk {c.symbol_name!r}"
                )


# ---------------------------------------------------------------------------
# SQL parsing
# ---------------------------------------------------------------------------

SQL_FUNCTIONS = """\
SET search_path TO platform;

CREATE OR REPLACE FUNCTION do_something(p_id BIGINT)
RETURNS VOID AS $$
BEGIN
    IF p_id IS NULL THEN
        RETURN;
    END IF;
    EXECUTE FORMAT('SELECT %L', p_id);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION do_another(p_name TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN UPPER(p_name);
END;
$$ LANGUAGE plpgsql;
"""

SQL_MIXED = """\
CREATE OR REPLACE FUNCTION get_user(p_id BIGINT)
RETURNS TABLE(name TEXT, email TEXT) AS $$
BEGIN
    RETURN QUERY SELECT u.name, u.email FROM users u WHERE u.id = p_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION delete_user(p_id BIGINT)
RETURNS VOID AS $$
BEGIN
    DELETE FROM users WHERE id = p_id;
END;
$$ LANGUAGE plpgsql;
"""

SQL_TRIGGERS = """\
CREATE OR REPLACE FUNCTION audit_handler()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_users_audit
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW EXECUTE FUNCTION audit_handler();
"""


class TestSQLParsing:
    """Tests for SQL-specific chunking and name extraction."""

    def test_function_name_extraction(self, parser):
        chunks = parser.parse_file("test.sql", SQL_FUNCTIONS)
        names = [c.symbol_name for c in chunks if c.chunk_type == "statement"]
        assert "do_something" in names
        assert "do_another" in names

    def test_multiple_functions(self, parser):
        """Should extract multiple CREATE FUNCTION statements."""
        chunks = parser.parse_file("test.sql", SQL_MIXED)
        stmt_chunks = [c for c in chunks if c.chunk_type == "statement"]
        names = [c.symbol_name for c in stmt_chunks if c.symbol_name]

        assert "get_user" in names
        assert "delete_user" in names

    def test_trigger_extraction(self, parser):
        """Should extract trigger functions and CREATE TRIGGER statements."""
        chunks = parser.parse_file("test.sql", SQL_TRIGGERS)
        names = [c.symbol_name for c in chunks if c.chunk_type == "statement"]
        assert "audit_handler" in names

    def test_sql_always_has_file_chunk(self, parser):
        """SQL files should always produce at least a file-level chunk."""
        chunks = parser.parse_file("test.sql", SQL_FUNCTIONS)
        file_chunks = [c for c in chunks if c.chunk_type == "file"]
        assert len(file_chunks) >= 1


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

# Each section needs > 30 tokens to pass min_tokens filter
MARKDOWN_DOC = """\
# Introduction

This is the introduction paragraph for the project documentation.
It describes the overall architecture and design goals of the system.
The platform supports multi-tenant deployments with row-level security
and metadata-driven schema management for dynamic object creation.

## Getting Started

To get started with the project, first install all required dependencies
using the package manager. Then configure your environment variables for
database connection, API keys, and tenant configuration. Run the setup
script to initialize the database schema and seed the platform tables.

## API Reference

The API provides CRUD operations for all platform objects. Each endpoint
supports filtering, pagination, and field selection. Authentication is
handled via JWT tokens with tenant-scoped permissions and role-based access.
"""


class TestMarkdownParsing:
    def test_section_extraction(self, parser):
        chunks = parser.parse_file("doc.md", MARKDOWN_DOC)
        section_names = [c.symbol_name for c in chunks if c.chunk_type == "section"]

        assert "Introduction" in section_names
        assert "Getting Started" in section_names
        assert "API Reference" in section_names

    def test_section_text_integrity(self, parser):
        chunks = parser.parse_file("doc.md", MARKDOWN_DOC)
        intro = next(c for c in chunks if c.symbol_name == "Introduction")
        assert "introduction paragraph" in intro.text


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_standard_extensions(self):
        from code_context.chunking.languages import detect_language

        cases = {
            "app.ts": "typescript",
            "component.tsx": "typescript",
            "index.js": "javascript",
            "main.py": "python",
            "schema.sql": "sql",
            "README.md": "markdown",
            "config.json": "json",
            "settings.yaml": "yaml",
            "Cargo.toml": "toml",
            "styles.css": "css",
            "page.html": "html",
            "Main.java": "java",
            "main.go": "go",
            "lib.rs": "rust",
        }
        for filepath, expected in cases.items():
            assert detect_language(filepath) == expected, f"{filepath} should be {expected}"

    def test_unknown_extension(self):
        from code_context.chunking.languages import detect_language

        assert detect_language("data.parquet") is None
        assert detect_language("Makefile") is None
        assert detect_language("image.png") is None

    def test_case_insensitive(self):
        from code_context.chunking.languages import detect_language

        assert detect_language("Schema.SQL") == "sql"
        assert detect_language("App.TS") == "typescript"


# ---------------------------------------------------------------------------
# Module re-exports (refactoring safety net)
# ---------------------------------------------------------------------------

class TestModuleReExports:
    """Verify that the module split doesn't break any import paths."""

    def test_parser_module_exports_all_legacy_symbols(self):
        """All symbols previously in parser.py must still be importable from it."""
        from code_context.chunking.parser import (
            CodeParser,
            ParsedChunk,
            compute_chunk_hash,
            compute_file_hash,
            detect_language,
        )
        assert CodeParser is not None
        assert ParsedChunk is not None
        assert callable(compute_chunk_hash)
        assert callable(compute_file_hash)
        assert callable(detect_language)

    def test_package_level_imports(self):
        from code_context.chunking import (
            CodeParser,
            ParsedChunk,
            compute_chunk_hash,
            compute_file_hash,
            detect_language,
            LANGUAGE_CONFIG,
        )
        assert isinstance(LANGUAGE_CONFIG, dict)
        assert len(LANGUAGE_CONFIG) > 0

    def test_direct_submodule_imports(self):
        from code_context.chunking.models import ParsedChunk, _encoder
        from code_context.chunking.languages import LANGUAGE_CONFIG, detect_language

        assert _encoder is not None
        assert "sql" in LANGUAGE_CONFIG
