#!/usr/bin/env python3
"""CLI for Code Context - uses gum for beautiful TUI interactions.

Commands:
    index <path>     - Index a project directory
    reindex <path>   - Force reindex a file or directory
    stats            - Show index statistics
    watch <path>     - Watch for file changes (real-time indexing)
    prune            - Remove orphaned chunks

Requires: gum (https://github.com/charmbracelet/gum)
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path


def gum_spin(title: str, command: list[str]) -> subprocess.CompletedProcess:
    """Run a command with gum spinner."""
    return subprocess.run(
        ["gum", "spin", "--spinner", "dot", "--title", title, "--"] + command,
        capture_output=True,
        text=True,
    )


def gum_style(text: str, **kwargs) -> str:
    """Style text with gum."""
    args = ["gum", "style"]
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key.replace('_', '-')}")
        else:
            args.extend([f"--{key.replace('_', '-')}", str(value)])
    args.append("--")  # Separator before positional argument
    args.append(text)
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else text


def gum_confirm(prompt: str) -> bool:
    """Show confirmation prompt with gum."""
    result = subprocess.run(
        ["gum", "confirm", prompt],
        capture_output=True,
    )
    return result.returncode == 0


def gum_choose(title: str, options: list[str]) -> str | None:
    """Show selection menu with gum."""
    print(gum_style(title, bold=True, foreground="212"))
    result = subprocess.run(
        ["gum", "choose"] + options,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def print_header():
    """Print CLI header."""
    header = gum_style(
        "Code Context CLI",
        bold=True,
        foreground="212",
        border="rounded",
        padding="1 4",
        margin="1",
    )
    print(header)


def print_success(message: str):
    """Print success message."""
    print(gum_style(f"✓ {message}", foreground="10"))


def print_error(message: str):
    """Print error message."""
    print(gum_style(f"✗ {message}", foreground="9"))


def print_info(message: str):
    """Print info message."""
    print(gum_style(f"→ {message}", foreground="14"))


async def cmd_index(args):
    """Index a project directory."""
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    path = Path(args.path).resolve()
    if not path.is_dir():
        print_error(f"Not a directory: {path}")
        return 1

    print_info(f"Indexing project: {path}")

    db = DatabasePool()
    await db.initialize()

    try:
        voyage = VoyageClient()
        indexer = Indexer(db, voyage)

        stats = await indexer.index_project(str(path), force=args.force)

        print()
        print_success(f"Indexed {stats['indexed_files']} files")
        print_info(f"Total files scanned: {stats['total_files']}")
        print_info(f"Files skipped (unchanged): {stats['skipped_files']}")
        print_info(f"Total chunks created: {stats['total_chunks']}")

        if stats["errors"]:
            print()
            print_error(f"Errors: {len(stats['errors'])}")
            for err in stats["errors"][:5]:
                print(f"  - {err['file']}: {err['error']}")

        return 0
    finally:
        await db.close()


async def cmd_reindex(args):
    """Force reindex a file or directory."""
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    path = Path(args.path).resolve()

    db = DatabasePool()
    await db.initialize()

    try:
        voyage = VoyageClient()
        indexer = Indexer(db, voyage)

        if path.is_dir():
            print_info(f"Force reindexing directory: {path}")
            stats = await indexer.index_project(str(path), force=True)
            print_success(f"Reindexed {stats['indexed_files']} files")
        elif path.is_file():
            print_info(f"Force reindexing file: {path}")
            result = await indexer.index_file(str(path), str(path.parent), force=True)
            if result["indexed"]:
                print_success(f"Reindexed: {result['chunks']} chunks")
            else:
                print_error(f"Could not index: {result.get('reason', 'unknown')}")
        else:
            print_error(f"Path not found: {path}")
            return 1

        return 0
    finally:
        await db.close()


async def cmd_stats(args):
    """Show index statistics."""
    from code_context.db.pool import DatabasePool

    db = DatabasePool()
    await db.initialize()

    try:
        stats = await db.get_index_stats()

        print()
        print(gum_style("Index Statistics", bold=True, foreground="212"))
        print()
        print(f"  Total files:    {stats['total_files']}")
        print(f"  Total chunks:   {stats['total_chunks']}")
        print(f"  Total projects: {stats['total_projects']}")
        print()

        if stats["by_language"]:
            print(gum_style("By Language:", bold=True))
            for lang, data in stats["by_language"].items():
                print(f"\n  {lang}:")
                print(f"    Files:     {data['files']}")
                print(f"    Chunks:    {data['chunks']}")
                print(f"    Functions: {data['functions']}")
                print(f"    Classes:   {data['classes']}")
                print(f"    Methods:   {data['methods']}")

        return 0
    finally:
        await db.close()


async def cmd_watch(args):
    """Watch for file changes and reindex in real-time."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print_error("watchdog not installed. Run: uv pip install watchdog")
        return 1

    from code_context.chunking.parser import detect_language
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    path = Path(args.path).resolve()
    if not path.is_dir():
        print_error(f"Not a directory: {path}")
        return 1

    db = DatabasePool()
    await db.initialize()
    voyage = VoyageClient()
    indexer = Indexer(db, voyage)

    class IndexHandler(FileSystemEventHandler):
        def __init__(self):
            self.loop = asyncio.get_event_loop()

        def on_modified(self, event):
            if event.is_directory:
                return
            if detect_language(event.src_path):
                print_info(f"File modified: {event.src_path}")
                asyncio.run_coroutine_threadsafe(
                    indexer.index_file(event.src_path, str(path)),
                    self.loop,
                )

        def on_created(self, event):
            if event.is_directory:
                return
            if detect_language(event.src_path):
                print_info(f"File created: {event.src_path}")
                asyncio.run_coroutine_threadsafe(
                    indexer.index_file(event.src_path, str(path)),
                    self.loop,
                )

        def on_deleted(self, event):
            if event.is_directory:
                return
            if detect_language(event.src_path):
                print_info(f"File deleted: {event.src_path}")
                asyncio.run_coroutine_threadsafe(
                    indexer.remove_file(event.src_path),
                    self.loop,
                )

    observer = Observer()
    observer.schedule(IndexHandler(), str(path), recursive=True)
    observer.start()

    print_success(f"Watching for changes in: {path}")
    print_info("Press Ctrl+C to stop")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print()
        print_info("Stopped watching")

    observer.join()
    await db.close()
    return 0


async def cmd_prune(args):
    """Remove orphaned chunks."""
    from code_context.db.pool import DatabasePool

    if not gum_confirm("Remove all orphaned chunks from the database?"):
        print_info("Cancelled")
        return 0

    db = DatabasePool()
    await db.initialize()

    try:
        async with db.acquire() as conn:
            result = await conn.fetchval("SELECT cleanup_orphaned_chunks()")
            print_success(f"Removed {result} orphaned chunks")
        return 0
    finally:
        await db.close()


async def cmd_index_books(args):
    """Index books from a directory."""
    from code_context.books.indexer import BookIndexer
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient

    path = Path(args.path).resolve()
    if not path.is_dir():
        print_error(f"Not a directory: {path}")
        return 1

    print_info(f"Indexing books from: {path}")

    db = DatabasePool()
    await db.initialize()

    try:
        voyage = VoyageClient()
        indexer = BookIndexer(db, voyage)

        results = await indexer.index_directory(
            path, force=args.force, category=args.category
        )

        print()
        indexed = 0
        skipped = 0
        for result in results:
            if "error" in result:
                print_error(f"Failed: {result['filepath']}: {result['error']}")
            elif result.get("skipped"):
                skipped += 1
            else:
                indexed += 1
                print_success(
                    f"{result['title']} ({result['book_id']}): "
                    f"{result['chunks_indexed']} chunks"
                )

        if skipped:
            print_info(f"Skipped {skipped} already-indexed book(s)")

        # Create vector index
        print_info("Creating vector index...")
        await indexer.create_vector_index()
        print_success("Vector index created")

        return 0
    finally:
        await db.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Code Context CLI - Indexing and maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # index command
    index_parser = subparsers.add_parser("index", help="Index a project directory")
    index_parser.add_argument("path", help="Project directory to index")
    index_parser.add_argument(
        "-f", "--force", action="store_true", help="Force reindex all files"
    )

    # reindex command
    reindex_parser = subparsers.add_parser(
        "reindex", help="Force reindex a file or directory"
    )
    reindex_parser.add_argument("path", help="File or directory to reindex")

    # stats command
    subparsers.add_parser("stats", help="Show index statistics")

    # watch command
    watch_parser = subparsers.add_parser(
        "watch", help="Watch for file changes (real-time indexing)"
    )
    watch_parser.add_argument("path", help="Directory to watch")

    # prune command
    subparsers.add_parser("prune", help="Remove orphaned chunks")

    # index-books command
    books_parser = subparsers.add_parser("index-books", help="Index books from a directory")
    books_parser.add_argument("path", help="Directory containing EPUB files")
    books_parser.add_argument(
        "-f", "--force", action="store_true", help="Force reindex all books"
    )
    books_parser.add_argument(
        "-c", "--category", default="software", help="Book category (default: software)"
    )

    args = parser.parse_args()

    if not args.command:
        print_header()
        parser.print_help()
        return 0

    print_header()

    # Run the appropriate command
    commands = {
        "index": cmd_index,
        "reindex": cmd_reindex,
        "stats": cmd_stats,
        "watch": cmd_watch,
        "prune": cmd_prune,
        "index-books": cmd_index_books,
    }

    cmd = commands.get(args.command)
    if cmd:
        return asyncio.run(cmd(args))
    else:
        print_error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
