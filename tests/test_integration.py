"""Integration tests for code-context-v2.

Run with: uv run pytest tests/ -v
"""

import json
import pytest

from code_context.chunking.parser import CodeParser, compute_chunk_hash, compute_file_hash
from code_context.db.pool import DatabasePool
from code_context.db.schema import CodeChunk, CodeFile


# Sample code for testing
TYPESCRIPT_CODE = '''
import { useState } from 'react';

export function validateEmail(email: string): boolean {
  return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+/.test(email);
}

export class UserService {
  private users: Map<string, User> = new Map();

  async getUser(id: string): Promise<User | null> {
    return this.users.get(id) || null;
  }

  async createUser(data: UserInput): Promise<User> {
    const user = { id: crypto.randomUUID(), ...data };
    this.users.set(user.id, user);
    return user;
  }
}
'''

PYTHON_CODE = '''
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    id: str
    name: str
    email: str

def validate_email(email: str) -> bool:
    """Validate an email address."""
    return "@" in email and "." in email

class UserRepository:
    def __init__(self):
        self._users = {}

    def get(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def save(self, user: User) -> None:
        self._users[user.id] = user
'''

LARGE_TYPESCRIPT_CODE = TYPESCRIPT_CODE + "\n" + "\n".join(
    [f"const pad_{i} = {i};" for i in range(230)]
)

LARGE_PYTHON_CODE = PYTHON_CODE + "\n" + "\n".join(
    [f"x_{i} = {i}" for i in range(230)]
)

# Large function that exceeds chunk_max_tokens (600) — many statements
LARGE_PYTHON_FUNCTION = '''
def process_data(items):
    """Process a large batch of items with multiple steps."""
''' + "\n".join(
    [f"    result_{i} = items[{i}] * {i + 1} + {i * 3}  # step {i}" for i in range(120)]
) + '''
    return locals()
'''

# Large class with multiple methods — should split into method-level chunks
LARGE_TS_CLASS = '''
import { Database } from 'pg';

export class OrderService {
  private db: Database;

  constructor(db: Database) {
    this.db = db;
  }

''' + "\n".join([
    f'''  async processOrder{i}(orderId: string): Promise<void> {{
    const order = await this.db.query('SELECT * FROM orders WHERE id = $1', [orderId]);
    const items = await this.db.query('SELECT * FROM items WHERE order_id = $1', [orderId]);
    const total = items.reduce((sum: number, item: any) => sum + item.price * item.quantity, 0);
    await this.db.query('UPDATE orders SET total = $1, status = $2 WHERE id = $3', [total, 'processed_{i}', orderId]);
    console.log(`Order ${{orderId}} processed with total ${{total}} for batch {i}`);
  }}
''' for i in range(8)
]) + '''
}
'''


class TestParser:
    """Test the code parser."""

    def setup_method(self):
        self.parser = CodeParser()

    def test_parse_typescript_function(self):
        """Small TS files should still expose symbol-level chunks."""
        chunks = self.parser.parse_file("test.ts", TYPESCRIPT_CODE)
        assert len(chunks) >= 2, f"Expected symbol chunks for small file, got {len(chunks)}"
        assert any(c.symbol_name == "validateEmail" and c.chunk_type == "function" for c in chunks)
        assert any(c.symbol_name == "UserService" and c.chunk_type == "declaration" for c in chunks)
        assert all("filepath" in c.context for c in chunks)
        assert all("imports" in c.context for c in chunks)
        assert all(c.chunk_type != "file" for c in chunks), "File chunk should be dropped when symbols exist"

    def test_parse_typescript_class(self):
        """Large TS files should include extracted symbols."""
        chunks = self.parser.parse_file("test.ts", LARGE_TYPESCRIPT_CODE)

        # validateEmail appears as function chunk in larger files
        validate_chunks = [c for c in chunks if c.symbol_name == "validateEmail"]
        assert validate_chunks, "Should find validateEmail symbol"
        assert any(c.chunk_type == "function" for c in validate_chunks)

        # UserService is extracted as declaration chunk
        class_chunks = [c for c in chunks if c.symbol_name == "UserService"]
        assert class_chunks, "Should find UserService symbol"
        assert any(c.chunk_type == "declaration" for c in class_chunks)
        assert any("export class UserService" in c.text for c in class_chunks)

    def test_parse_python_function(self):
        """Small Python files should prefer declarations over generic file chunks."""
        chunks = self.parser.parse_file("test.py", PYTHON_CODE)
        assert len(chunks) >= 1, f"Expected at least one symbol chunk, got {len(chunks)}"
        assert any(c.symbol_name == "UserRepository" and c.chunk_type == "declaration" for c in chunks)
        assert all(c.chunk_type != "file" for c in chunks), "File chunk should be dropped when symbols exist"

    def test_parse_python_class(self):
        """Large Python files should expose declarations with symbol names."""
        chunks = self.parser.parse_file("test.py", LARGE_PYTHON_CODE)

        # Small Python function is filtered out by chunk_min_tokens
        func_chunk = next((c for c in chunks if c.symbol_name == "validate_email"), None)
        assert func_chunk is None, "validate_email should be filtered as too small"

        class_chunk = next((c for c in chunks if c.symbol_name == "UserRepository"), None)
        assert class_chunk is not None, "Should find UserRepository declaration"
        assert class_chunk.chunk_type == "declaration"

    def test_context_metadata_is_dict(self):
        """Test that context metadata is a dict (critical for JSONB)."""
        chunks = self.parser.parse_file("test.ts", TYPESCRIPT_CODE)

        for chunk in chunks:
            assert isinstance(chunk.context, dict), f"Context should be dict, got {type(chunk.context)}"
            # Should be JSON serializable
            json_str = json.dumps(chunk.context)
            assert isinstance(json_str, str)

    def test_compute_hashes(self):
        """Test hash computation."""
        content = "test content"
        file_hash = compute_file_hash(content)
        assert len(file_hash) == 64, "BLAKE3 hash should be 64 hex chars"

        chunk_hash = compute_chunk_hash("chunk text", "/path/file.ts", 10)
        assert len(chunk_hash) == 32, "Chunk hash should be 32 hex chars"

    def test_large_function_split_preserves_statement_boundaries(self):
        """Large Python function should split at statement boundaries, not mid-statement."""
        chunks = self.parser.parse_file("test_large_func.py", LARGE_PYTHON_FUNCTION)

        # Should produce multiple chunks from the large function
        func_chunks = [c for c in chunks if c.chunk_type in ("function", "declaration")]
        assert len(func_chunks) >= 2, (
            f"Large function should be split into >=2 chunks, got {len(func_chunks)}"
        )

        # Each chunk should contain complete lines (no mid-line breaks)
        for chunk in func_chunks:
            lines = chunk.text.split("\n")
            for line in lines:
                # Lines should not start with partial tokens (encoding artifacts)
                # A valid Python line either is empty, starts with whitespace+identifier,
                # or starts with a keyword/comment
                assert not line or line[0] in " \t#)]}\"'" or line[0].isalpha() or line[0] == "@", (
                    f"Chunk contains likely mid-line split: {line[:50]!r}"
                )

    def test_large_class_split_into_methods(self):
        """Large TS class should split structurally into method-level chunks."""
        chunks = self.parser.parse_file("test_large_class.ts", LARGE_TS_CLASS)

        # Should have method-level chunks
        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        assert len(method_chunks) >= 3, (
            f"Large class should produce method chunks, got {len(method_chunks)}"
        )

        # Each method chunk should contain a complete method signature
        for chunk in method_chunks:
            assert "async " in chunk.text or "processOrder" in chunk.text, (
                f"Method chunk should contain method code: {chunk.text[:80]}"
            )

    def test_line_boundary_fallback(self):
        """Chunk without AST children should fall back to line-boundary splitting."""
        # Create a large comment block (no structural AST children)
        big_comment = "# " + "\n# ".join([f"Comment line {i} with some padding text here" for i in range(200)])
        code = big_comment + "\nx = 1\n"

        chunks = self.parser.parse_file("test_comment.py", code)
        # The file chunk (if split) should split at line boundaries
        for chunk in chunks:
            if chunk.chunk_type == "file" or "_part" in (chunk.symbol_name or ""):
                lines = chunk.text.split("\n")
                for line in lines:
                    # No line should be a partial encoding artifact
                    assert not line or line[0] in " \t#x=" or line[0].isalpha(), (
                        f"Line-split produced artifact: {line[:50]!r}"
                    )

    def test_signature_in_context(self):
        """Function chunks should have signature in context metadata."""
        chunks = self.parser.parse_file("test_sig.py", PYTHON_CODE)
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        for chunk in func_chunks:
            sig = chunk.context.get("signature")
            if sig:
                assert "def " in sig, f"Signature should contain 'def': {sig}"
                assert "{" not in sig and ":" in sig, f"Signature should not contain body: {sig}"

    def test_ast_split_disabled_falls_back(self, monkeypatch):
        """When ast_split_enabled=False, should use old token-based splitting."""
        from code_context.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "ast_split_enabled", False)

        chunks = self.parser.parse_file("test_fallback.py", LARGE_PYTHON_FUNCTION)
        # Should still produce chunks (via token split fallback)
        assert len(chunks) >= 1, "Should produce at least one chunk even with token split"


class TestDatabasePool:
    """Test database operations."""

    @pytest.fixture
    def db(self):
        """Create database pool for tests."""
        return DatabasePool()

    @pytest.mark.asyncio
    async def test_connection(self, db):
        """Test database connection."""
        await db.initialize()
        try:
            async with db.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                assert result == 1
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_file_operations(self, db):
        """Test file CRUD operations."""
        from datetime import datetime

        await db.initialize()
        try:
            test_file = CodeFile(
                filepath="/test/integration_test.ts",
                file_hash="abc123",
                language="typescript",
                project_root="/test",
                project_id="test",
                last_indexed=datetime.now(),
                loc=100,
            )

            # Insert
            await db.upsert_file(test_file)

            # Read
            retrieved = await db.get_file("/test/integration_test.ts")
            assert retrieved is not None
            assert retrieved.filepath == test_file.filepath
            assert retrieved.language == "typescript"

            # Delete
            await db.delete_file("/test/integration_test.ts")
            retrieved = await db.get_file("/test/integration_test.ts")
            assert retrieved is None
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_chunk_operations_with_jsonb(self, db):
        """Test chunk operations with JSONB metadata (critical test for the fix)."""
        from datetime import datetime

        await db.initialize()
        try:
            # Cleanup any existing test data first
            await db.delete_file("/test/jsonb_test.ts")

            # First create a file
            test_file = CodeFile(
                filepath="/test/jsonb_test.ts",
                file_hash="def456",
                language="typescript",
                project_root="/test",
                project_id="test",
                last_indexed=datetime.now(),
                loc=50,
            )
            await db.upsert_file(test_file)

            # Create chunks with context_metadata as dict (this is what caused the bug)
            test_chunks = [
                CodeChunk(
                    filepath="/test/jsonb_test.ts",
                    chunk_hash="chunk1",
                    chunk_text="function test() { return true; }",
                    embedding=[0.1] * 1024,  # Mock embedding
                    start_line=1,
                    end_line=3,
                    chunk_type="function",
                    symbol_name="test",
                    context_metadata={
                        "filepath": "/test/jsonb_test.ts",
                        "imports": ["import { something } from 'somewhere'"],
                        "parent_class": None,
                    },
                    token_count=10,
                ),
                CodeChunk(
                    filepath="/test/jsonb_test.ts",
                    chunk_hash="chunk2",
                    chunk_text="class MyClass { method() {} }",
                    embedding=[0.2] * 1024,
                    start_line=5,
                    end_line=10,
                    chunk_type="class",
                    symbol_name="MyClass",
                    context_metadata={
                        "filepath": "/test/jsonb_test.ts",
                        "imports": [],
                    },
                    token_count=8,
                ),
            ]

            # This should NOT raise an error after the fix
            await db.insert_chunks(test_chunks)

            # Verify chunks were inserted
            async with db.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM code_chunks WHERE filepath = $1",
                    "/test/jsonb_test.ts"
                )
                assert count == 2, f"Expected 2 chunks, got {count}"

                # Verify JSONB data is correct
                # Note: asyncpg returns JSONB as dict when using set_type_codec, otherwise as string
                row = await conn.fetchrow(
                    "SELECT context_metadata FROM code_chunks WHERE chunk_hash = $1",
                    "chunk1"
                )
                metadata = row["context_metadata"]
                # Handle both dict (with codec) and string (without codec) returns
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                assert isinstance(metadata, dict), f"Metadata should be dict, got {type(metadata)}"
                assert metadata["filepath"] == "/test/jsonb_test.ts"

            # Cleanup
            await db.delete_file("/test/jsonb_test.ts")
        finally:
            await db.close()


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_parse_and_store(self):
        """Test parsing code and storing in database."""
        from datetime import datetime

        parser = CodeParser()
        db = DatabasePool()

        await db.initialize()
        try:
            # Cleanup any existing test data first
            await db.delete_file("/test/e2e_test.ts")

            # Parse TypeScript code
            chunks = parser.parse_file("e2e_test.ts", TYPESCRIPT_CODE)
            assert len(chunks) > 0

            # Create file record
            test_file = CodeFile(
                filepath="/test/e2e_test.ts",
                file_hash=compute_file_hash(TYPESCRIPT_CODE),
                language="typescript",
                project_root="/test",
                project_id="test",
                last_indexed=datetime.now(),
                loc=TYPESCRIPT_CODE.count("\n") + 1,
            )
            await db.upsert_file(test_file)

            # Create chunk objects (simulating what indexer does)
            db_chunks = [
                CodeChunk(
                    filepath="/test/e2e_test.ts",
                    chunk_hash=compute_chunk_hash(chunk.text, "/test/e2e_test.ts", chunk.start_line),
                    chunk_text=chunk.text,
                    embedding=[0.1] * 1024,  # Mock embedding
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type,
                    symbol_name=chunk.symbol_name,
                    context_metadata=chunk.context,  # This is a dict
                    token_count=chunk.token_count,
                )
                for chunk in chunks
            ]

            # Store chunks (this tests the JSONB fix)
            await db.insert_chunks(db_chunks)

            # Verify
            async with db.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM code_chunks WHERE filepath = $1",
                    "/test/e2e_test.ts"
                )
                assert count == len(chunks), f"Expected {len(chunks)} chunks, got {count}"

            # Cleanup
            await db.delete_file("/test/e2e_test.ts")
        finally:
            await db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
