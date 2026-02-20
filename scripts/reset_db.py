#!/usr/bin/env python3
"""Reset database - clears all indexed data.

Run with: uv run python scripts/reset_db.py
"""

import asyncio
import sys
sys.path.insert(0, "src")

from code_context.db.pool import DatabasePool


async def reset_database():
    """Clear all data from code_files and code_chunks tables."""
    db = DatabasePool()
    await db.initialize()

    try:
        async with db.acquire() as conn:
            # Get counts before
            files_before = await conn.fetchval("SELECT COUNT(*) FROM code_files")
            chunks_before = await conn.fetchval("SELECT COUNT(*) FROM code_chunks")
            print(f"Before: {files_before} files, {chunks_before} chunks")

            # Delete all data (chunks cascade from files)
            await conn.execute("DELETE FROM code_files")

            # Verify
            files_after = await conn.fetchval("SELECT COUNT(*) FROM code_files")
            chunks_after = await conn.fetchval("SELECT COUNT(*) FROM code_chunks")
            print(f"After: {files_after} files, {chunks_after} chunks")

            print("Database reset complete!")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(reset_database())
