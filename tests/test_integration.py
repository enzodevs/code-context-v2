"""Integration tests for code-context-v2.

Run with: uv run pytest tests/ -v
"""

import asyncio
import json
import pytest
from pathlib import Path

from code_context.chunking.parser import CodeParser, compute_chunk_hash, compute_file_hash
from code_context.db.pool import DatabasePool
from code_context.db.schema import CodeChunk, CodeFile
from code_context.config import get_settings


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


class TestParser:
    """Test the code parser."""

    def setup_method(self):
        self.parser = CodeParser()

    def test_parse_typescript_function(self):
        """Test parsing TypeScript exported function."""
        chunks = self.parser.parse_file("test.ts", TYPESCRIPT_CODE)

        # Should find: validateEmail function, UserService class, getUser method, createUser method
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

        # Find the validateEmail function
        func_chunk = next((c for c in chunks if c.symbol_name == "validateEmail"), None)
        assert func_chunk is not None, "Should find validateEmail function"
        assert func_chunk.chunk_type == "function"
        assert "export function validateEmail" in func_chunk.text

        # Check context metadata
        assert "filepath" in func_chunk.context
        assert "imports" in func_chunk.context

    def test_parse_typescript_class(self):
        """Test parsing TypeScript exported class."""
        chunks = self.parser.parse_file("test.ts", TYPESCRIPT_CODE)

        # Find the UserService class
        class_chunk = next((c for c in chunks if c.symbol_name == "UserService"), None)
        assert class_chunk is not None, "Should find UserService class"
        assert class_chunk.chunk_type == "class"
        assert "export class UserService" in class_chunk.text

    def test_parse_python_function(self):
        """Test parsing Python function."""
        chunks = self.parser.parse_file("test.py", PYTHON_CODE)

        # Find the validate_email function
        func_chunk = next((c for c in chunks if c.symbol_name == "validate_email"), None)
        assert func_chunk is not None, "Should find validate_email function"
        assert func_chunk.chunk_type == "function"

    def test_parse_python_class(self):
        """Test parsing Python class."""
        chunks = self.parser.parse_file("test.py", PYTHON_CODE)

        # Find the UserRepository class
        class_chunk = next((c for c in chunks if c.symbol_name == "UserRepository"), None)
        assert class_chunk is not None, "Should find UserRepository class"
        assert class_chunk.chunk_type == "class"

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
