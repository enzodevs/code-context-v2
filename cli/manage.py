#!/usr/bin/env python3
"""Interactive management CLI for Code Context using gum TUI.

Run with: uv run python cli/manage.py
Requires: gum (https://github.com/charmbracelet/gum)
"""

import asyncio
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def gum_available() -> bool:
    """Check if gum is installed."""
    try:
        subprocess.run(["gum", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def gum_style(text: str, **kwargs) -> str:
    """Style text with gum."""
    args = ["gum", "style"]
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key.replace('_', '-')}")
        else:
            args.extend([f"--{key.replace('_', '-')}", str(value)])
    args.append("--")
    args.append(text)
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else text


def gum_choose(options: list[str], header: str = "", height: int = 0) -> str | None:
    """Show selection menu with gum."""
    args = [
        "gum", "choose",
        "--cursor", "▸ ",
        "--cursor.foreground", "212",
        "--header.foreground", "99",
    ]
    if header:
        args.extend(["--header", header])
    if height > 0:
        args.extend(["--height", str(height)])
    args.append("--")
    args.extend(options)
    result = subprocess.run(args, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def gum_filter(options: list[str], header: str = "", placeholder: str = "Type to filter...") -> str | None:
    """Show filterable list with gum.

    For small lists (<=8 items), falls back to gum choose (arrow-key selection).
    For larger lists, uses gum filter with fuzzy search.
    """
    if len(options) <= 8:
        return gum_choose(options, header=header)

    args = [
        "gum", "filter",
        "--indicator.foreground", "212",
        "--prompt.foreground", "99",
        "--placeholder", placeholder,
        "--header.foreground", "99",
    ]
    if header:
        args.extend(["--header", header])
    input_text = "\n".join(options)
    result = subprocess.run(args, input=input_text, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def gum_confirm(prompt: str) -> bool:
    """Show confirmation prompt with gum."""
    result = subprocess.run([
        "gum", "confirm", prompt,
        "--affirmative", "Yes", "--negative", "No",
        "--prompt.foreground", "212",
        "--selected.background", "212",
        "--selected.foreground", "230",
    ])
    return result.returncode == 0


def gum_input(placeholder: str = "", header: str = "", value: str = "") -> str | None:
    """Get text input with gum."""
    args = [
        "gum", "input",
        "--cursor.foreground", "212",
        "--prompt.foreground", "99",
        "--prompt", "› ",
        "--header.foreground", "99",
    ]
    if header:
        args.extend(["--header", header])
    if placeholder:
        args.extend(["--placeholder", placeholder])
    if value:
        args.extend(["--value", value])
    result = subprocess.run(args, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def gum_spin(title: str, command: list[str]) -> subprocess.CompletedProcess:
    """Run command with spinner."""
    return subprocess.run(
        ["gum", "spin", "--spinner", "dot", "--title", title, "--show-output", "--"] + command,
        capture_output=True,
        text=True,
    )


def print_header():
    """Print CLI header."""
    header = gum_style(
        "  Code Context v2  ",
        bold=True,
        foreground="212",
        border="rounded",
        border_foreground="99",
        padding="0 2",
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


def print_warning(message: str):
    """Print warning message."""
    print(gum_style(f"⚠ {message}", foreground="11"))


def pause():
    """Styled pause prompt."""
    input("\033[90m  press enter to continue…\033[0m ")


async def list_projects():
    """List all indexed projects."""
    from code_context.db.pool import DatabasePool

    db = DatabasePool()
    await db.initialize()

    try:
        projects = await db.list_projects()

        if not projects:
            print_warning("No projects indexed yet.")
            print_info("Use 'Index new project' to add one.")
            return

        import csv
        import io

        home = str(Path.home())
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["ID", "Path", "Files", "Chunks", "LOC", "Languages", "Last Indexed"])
        for p in projects:
            langs = ", ".join(p["languages"]) if p["languages"] else "—"
            last = p["last_indexed"].strftime("%Y-%m-%d %H:%M") if p["last_indexed"] else "never"
            path = p["project_root"].replace(home, "~")
            writer.writerow([
                p["project_id"], path, p["file_count"],
                p["chunk_count"], p["total_loc"], langs, last,
            ])

        print()
        subprocess.run(
            ["gum", "table", "--print",
             "--border", "rounded",
             "--border.foreground", "240",
             "--header.foreground", "212"],
            input=buf.getvalue(), text=True,
        )

    finally:
        await db.close()


async def view_project_stats():
    """View detailed stats for a specific project."""
    from code_context.db.pool import DatabasePool

    db = DatabasePool()
    await db.initialize()

    try:
        projects = await db.list_projects()

        if not projects:
            print_warning("No projects indexed yet.")
            return

        # Let user select a project by ID
        options = [f"{p['project_id']} ({p['project_root']})" for p in projects]
        selected = gum_filter(options, header="Select project to view stats:", placeholder="Type to filter...")

        if not selected:
            print_info("Cancelled.")
            return

        # Extract project_id from selection
        project_id = selected.split(" (")[0]
        stats = await db.get_project_stats(project_id)

        if not stats:
            print_error(f"Project not found: {project_id}")
            return

        print()
        print(gum_style(f"Stats for: {stats['project_id']}", bold=True, foreground="212"))
        print(f"  Path: {stats['project_root']}")
        print()
        print(f"  Total files:  {stats['file_count']}")
        print(f"  Total chunks: {stats['chunk_count']}")
        print(f"  Total LOC:    {stats['total_loc']}")
        if stats["last_indexed"]:
            print(f"  Last indexed: {stats['last_indexed'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        if stats["by_language"]:
            print(gum_style("By Language:", bold=True))
            for lang, data in stats["by_language"].items():
                print(f"\n  {gum_style(lang, foreground='14')}:")
                print(f"    Files:        {data['files']}")
                print(f"    Chunks:       {data['chunks']}")
                print(f"    ├─ file:        {data.get('files_chunks', 0)}")
                print(f"    ├─ declaration: {data.get('declarations', 0)}")
                print(f"    ├─ function:    {data['functions']}")
                print(f"    ├─ class:       {data['classes']}")
                print(f"    └─ method:      {data['methods']}")

    finally:
        await db.close()


async def delete_project():
    """Delete a specific project from the index."""
    from code_context.db.pool import DatabasePool

    db = DatabasePool()
    await db.initialize()

    try:
        projects = await db.list_projects()

        if not projects:
            print_warning("No projects indexed yet.")
            return

        # Let user select a project by ID
        options = [f"{p['project_id']} ({p['file_count']} files, {p['chunk_count']} chunks)" for p in projects]
        selected = gum_filter(options, header="Select project to DELETE:", placeholder="Type to filter...")

        if not selected:
            print_info("Cancelled.")
            return

        # Extract project_id from selection
        project_id = selected.split(" (")[0]

        # Confirm deletion
        print()
        print_warning(f"This will delete ALL indexed data for project: {project_id}")

        if not gum_confirm(f"Delete project '{project_id}'?"):
            print_info("Cancelled.")
            return

        result = await db.delete_project(project_id)
        print()
        print_success(f"Deleted {result['deleted_files']} files and {result['deleted_chunks']} chunks")

    finally:
        await db.close()


async def index_project():
    """Index a new project."""
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    # Step 1: Get project path
    path_input = gum_input(
        placeholder="/path/to/project",
        header="📁 Project path:"
    )

    if not path_input:
        print_info("Cancelled.")
        return

    # Expand ~ and resolve path
    project_path = Path(path_input).expanduser().resolve()

    if not project_path.is_dir():
        print_error(f"Not a directory: {project_path}")
        return

    # Step 2: Get project ID (suggest basename as default)
    default_id = project_path.name

    project_id = gum_input(
        header="🏷️  Project ID (short name for searches):",
        value=default_id,
    )

    # Use default if empty
    if not project_id:
        project_id = default_id

    # Validate project ID (no spaces, lowercase)
    project_id = project_id.strip().lower().replace(" ", "-")

    if not project_id:
        print_error("Project ID cannot be empty")
        return

    # Step 3: Confirm before indexing
    print()
    print(gum_style("Summary:", bold=True))
    print(f"  Path: {project_path}")
    print(f"  ID:   {project_id}")
    print()

    if not gum_confirm("Proceed with indexing?"):
        print_info("Cancelled.")
        return

    db = DatabasePool()
    await db.initialize()

    try:
        # Step 4: Check if project already exists, only ask force if it does
        force = False
        existing_projects = await db.list_projects()
        project_exists = any(p["project_id"] == project_id for p in existing_projects)

        if project_exists:
            force = gum_confirm("Project already indexed. Force reindex all files (even unchanged)?")

        print()
        print_info(f"Indexing: {project_path} as '{project_id}'")
        print_info("This may take a while for large projects...")
        print()
        voyage = VoyageClient()
        indexer = Indexer(db, voyage)

        stats = await indexer.index_project(
            project_root=str(project_path),
            project_id=project_id,
            force=force,
        )

        print()
        print_success(f"Indexed {stats['indexed_files']} files as '{project_id}'")
        print_info(f"Total files scanned: {stats['total_files']}")
        print_info(f"Files skipped (unchanged): {stats['skipped_files']}")
        print_info(f"Total chunks created: {stats['total_chunks']}")

        if stats["errors"]:
            print()
            print_warning(f"Errors: {len(stats['errors'])}")
            for err in stats["errors"][:5]:
                print(f"  - {err['file']}: {err['error']}")
            if len(stats["errors"]) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more")

    finally:
        await db.close()


async def sync_project():
    """Sync an existing project (reindex only changed files)."""
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    db = DatabasePool()
    await db.initialize()

    try:
        projects = await db.list_projects()

        if not projects:
            print_warning("No projects indexed yet.")
            print_info("Use 'Index new project' to add one first.")
            return

        # Let user select a project
        options = []
        for p in projects:
            last = p["last_indexed"].strftime("%Y-%m-%d %H:%M") if p["last_indexed"] else "never"
            options.append(f"{p['project_id']} | {p['file_count']} files | Last: {last}")

        selected = gum_filter(options, header="Select project to sync:", placeholder="Type to filter...")

        if not selected:
            print_info("Cancelled.")
            return

        # Extract project_id from selection
        project_id = selected.split(" | ")[0]
        project = next((p for p in projects if p["project_id"] == project_id), None)

        if not project:
            print_error(f"Project not found: {project_id}")
            return

        project_path = project["project_root"]

        print()
        print_info(f"Syncing: {project_path}")
        print_info("Checking for modified files...")
        print()

        voyage = VoyageClient()
        indexer = Indexer(db, voyage)

        stats = await indexer.index_project(
            project_root=project_path,
            project_id=project_id,
            force=False,  # Only changed files
        )

        print()
        deleted = stats.get("deleted_files", 0)
        if stats["indexed_files"] == 0 and deleted == 0:
            print_success("Already up to date! No changes detected.")
        else:
            parts = []
            if stats["indexed_files"] > 0:
                parts.append(f"{stats['indexed_files']} reindexed")
            if deleted > 0:
                parts.append(f"{deleted} removed")
            print_success(f"Synced: {', '.join(parts)}")

        print()
        print(f"  📁 Files scanned:    {stats['total_files']}")
        print(f"  ✏️  Files reindexed:  {stats['indexed_files']}")
        print(f"  ⏭️  Files unchanged:  {stats['skipped_files']}")
        if deleted > 0:
            print(f"  🗑️  Files removed:   {deleted}")
        print(f"  📦 Chunks created:   {stats['total_chunks']}")

        if stats["errors"]:
            print()
            print_warning(f"Errors: {len(stats['errors'])}")
            for err in stats["errors"][:5]:
                print(f"  - {err['file']}: {err['error']}")
            if len(stats["errors"]) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more")

    finally:
        await db.close()


async def check_sync_status():
    """Check if a project needs syncing (dry-run, no changes made)."""
    from code_context.db.pool import DatabasePool
    from code_context.indexer import Indexer

    db = DatabasePool()
    await db.initialize()

    try:
        projects = await db.list_projects()

        if not projects:
            print_warning("No projects indexed yet.")
            return

        # Let user select a project
        options = []
        for p in projects:
            last = p["last_indexed"].strftime("%Y-%m-%d %H:%M") if p["last_indexed"] else "never"
            options.append(f"{p['project_id']} | {p['file_count']} files | Last: {last}")

        selected = gum_filter(options, header="Select project to check:", placeholder="Type to filter...")

        if not selected:
            print_info("Cancelled.")
            return

        project_id = selected.split(" | ")[0]
        project = next((p for p in projects if p["project_id"] == project_id), None)

        if not project:
            print_error(f"Project not found: {project_id}")
            return

        project_path = project["project_root"]

        print()
        print_info(f"Checking: {project_path}")
        print_info("Comparing disk vs index (no changes will be made)...")
        print()

        indexer = Indexer(db, None)  # No VoyageClient needed for check
        status = await indexer.check_status(
            project_root=project_path,
            project_id=project_id,
        )

        _print_sync_status(status, project_path)

    finally:
        await db.close()


def _print_sync_status(status: dict, project_path: str):
    """Print sync status report."""
    root = Path(project_path)

    if status["up_to_date"]:
        print_success("Up to date! No changes detected.")
        print()
        print(f"  Files on disk:  {status['total_on_disk']}")
        print(f"  Files indexed:  {status['total_indexed']}")
        return

    print_warning("Out of sync! Changes detected:")
    print()

    if status["new"]:
        print(gum_style(f"  + {len(status['new'])} new file(s):", foreground="10"))
        for f in status["new"][:10]:
            try:
                rel = str(Path(f).relative_to(root))
            except ValueError:
                rel = f
            print(f"    + {rel}")
        if len(status["new"]) > 10:
            print(f"    ... and {len(status['new']) - 10} more")
        print()

    if status["modified"]:
        print(gum_style(f"  ~ {len(status['modified'])} modified file(s):", foreground="11"))
        for f in status["modified"][:10]:
            try:
                rel = str(Path(f).relative_to(root))
            except ValueError:
                rel = f
            print(f"    ~ {rel}")
        if len(status["modified"]) > 10:
            print(f"    ... and {len(status['modified']) - 10} more")
        print()

    if status["deleted"]:
        print(gum_style(f"  - {len(status['deleted'])} deleted file(s):", foreground="9"))
        for f in status["deleted"][:10]:
            try:
                rel = str(Path(f).relative_to(root))
            except ValueError:
                rel = f
            print(f"    - {rel}")
        if len(status["deleted"]) > 10:
            print(f"    ... and {len(status['deleted']) - 10} more")
        print()

    print(f"  Files on disk:    {status['total_on_disk']}")
    print(f"  Files indexed:    {status['total_indexed']}")
    print(f"  Unchanged:        {len(status['unchanged'])}")
    total_changes = len(status["new"]) + len(status["modified"]) + len(status["deleted"])
    print(f"  Total changes:    {total_changes}")
    if status.get("skipped_ineligible", 0) > 0:
        print(f"  Skipped (too large): {status['skipped_ineligible']}")
    print()
    print_info("Run 'Sync project' to apply these changes.")


async def watch_project():
    """Start a background watcher for a project."""
    from cli.watcher import list_active_watchers, start_watcher_daemon
    from code_context.db.pool import DatabasePool

    db = DatabasePool()
    await db.initialize()

    try:
        projects = await db.list_projects()
        active_watchers = list_active_watchers()
        watching_paths = {w["project_path"] for w in active_watchers}

        if projects:
            # Mark projects that are already being watched
            options = ["[Enter new path]"]
            for p in projects:
                if p["project_root"] in watching_paths:
                    options.append(f"{p['project_root']} [WATCHING]")
                else:
                    options.append(p["project_root"])

            selected = gum_choose(options, header="Select project to watch:")

            if not selected:
                print_info("Cancelled.")
                return

            if selected == "[Enter new path]":
                path_input = gum_input(placeholder="/path/to/project", header="Enter project path:")
                if not path_input:
                    print_info("Cancelled.")
                    return
                project_path = Path(path_input).expanduser().resolve()
            elif "[WATCHING]" in selected:
                print_warning("This project is already being watched.")
                return
            else:
                project_path = Path(selected)
        else:
            path_input = gum_input(placeholder="/path/to/project", header="Enter project path to watch:")
            if not path_input:
                print_info("Cancelled.")
                return
            project_path = Path(path_input).expanduser().resolve()

        if not project_path.is_dir():
            print_error(f"Not a directory: {project_path}")
            return

        await db.close()

        print()
        print_info(f"Starting background watcher for: {project_path}")

        success, message = start_watcher_daemon(str(project_path))

        if success:
            print_success(message)
            print_info("Watcher is running in background. Use 'Manage watchers' to stop it.")
        else:
            print_error(message)

    except Exception as e:
        print_error(f"Error: {e}")
        await db.close()


async def watch_all_projects():
    """Start the global watcher daemon (interactive)."""
    from cli.watcher import is_global_watcher_running, start_global_watcher_daemon

    if is_global_watcher_running():
        print_warning("Global watcher is already running.")
        return

    print_info("Starting global watcher (all indexed projects)...")
    success, message = start_global_watcher_daemon()
    if success:
        print_success(message)
        print_info("Use 'Manage watchers' to stop it.")
    else:
        print_error(message)


async def manage_watchers():
    """List and manage active watchers."""
    from cli.watcher import get_global_watcher_log, get_watcher_log, list_active_watchers, stop_watcher_by_pid

    while True:
        watchers = list_active_watchers()

        if not watchers:
            print_warning("No active watchers.")
            return

        print()

        options = []
        for w in watchers:
            started = w.get("started_at", "unknown")
            if isinstance(started, str) and "T" in started:
                started = started.split("T")[1].split(".")[0]
            if w.get("is_global"):
                options.append(f"PID {w['pid']} | [GLOBAL WATCHER] | Started: {started}")
            else:
                options.append(f"PID {w['pid']} | {Path(w['project_path']).name} | Started: {started}")

        options.append("────────────────────────────────")
        options.append("🛑 Stop all watchers")
        options.append("📜 View watcher logs")
        options.append("🔙 Back to main menu")

        selected = gum_choose(options, header=f"Active Watchers ({len(watchers)})")

        if not selected or "Back to main menu" in selected:
            return

        if "Stop all watchers" in selected:
            if gum_confirm("Stop all active watchers?"):
                for w in watchers:
                    stop_watcher_by_pid(w["pid"])
                print_success("All watchers stopped.")
            return

        if "View watcher logs" in selected:
            # Show log selection
            log_options = []
            for w in watchers:
                if w.get("is_global"):
                    log_file = get_global_watcher_log()
                    if log_file:
                        log_options.append(f"[GLOBAL] → {log_file}")
                else:
                    log_file = get_watcher_log(w["project_path"])
                    if log_file:
                        log_options.append(f"{Path(w['project_path']).name} → {log_file}")

            if not log_options:
                print_warning("No log files found.")
                continue

            log_selected = gum_choose(log_options, header="Select log to view:")
            if log_selected:
                log_path = log_selected.split(" → ")[1]
                print()
                print(gum_style(f"Log: {log_path}", bold=True))
                print()
                # Show last 30 lines
                subprocess.run(["tail", "-30", log_path])
                pause()
            continue

        if selected.startswith("─"):
            continue

        # Stop specific watcher
        pid = int(selected.split()[1])
        watcher = next((w for w in watchers if w["pid"] == pid), None)

        if watcher:
            print()
            print_info(f"Project: {watcher['project_path']}")
            print_info(f"PID: {pid}")
            print_info(f"Started: {watcher.get('started_at', 'unknown')}")

            if gum_confirm("Stop this watcher?"):
                if stop_watcher_by_pid(pid):
                    print_success("Watcher stopped.")
                else:
                    print_error("Failed to stop watcher.")


async def prune_orphans():
    """Remove orphaned chunks from the database."""
    from code_context.db.pool import DatabasePool

    if not gum_confirm("Remove all orphaned chunks from the database?"):
        print_info("Cancelled.")
        return

    db = DatabasePool()
    await db.initialize()

    try:
        async with db.acquire() as conn:
            result = await conn.fetchval("SELECT cleanup_orphaned_chunks()")
            print_success(f"Removed {result} orphaned chunks")
    finally:
        await db.close()


async def reset_database():
    """Reset entire database (delete all data)."""
    from code_context.db.pool import DatabasePool

    print()
    print_warning("This will DELETE ALL indexed data!")
    print()

    if not gum_confirm("Are you sure you want to reset the database?"):
        print_info("Cancelled.")
        return

    # Double confirm for destructive action
    if not gum_confirm("This action cannot be undone. Continue?"):
        print_info("Cancelled.")
        return

    db = DatabasePool()
    await db.initialize()

    try:
        async with db.acquire() as conn:
            files_before = await conn.fetchval("SELECT COUNT(*) FROM code_files")
            chunks_before = await conn.fetchval("SELECT COUNT(*) FROM code_chunks")

            await conn.execute("DELETE FROM code_files")

            print()
            print_success(f"Deleted {files_before} files and {chunks_before} chunks")
            print_info("Database is now empty.")
    finally:
        await db.close()


async def global_stats():
    """Show global statistics across all projects."""
    from code_context.db.pool import DatabasePool

    db = DatabasePool()
    await db.initialize()

    try:
        stats = await db.get_index_stats()
        quality = await db.get_quality_stats()

        print()
        print(gum_style("Global Statistics", bold=True, foreground="212"))
        print()
        print(f"  Total files:    {stats['total_files']}")
        print(f"  Total chunks:   {stats['total_chunks']}")
        print(f"  Total projects: {stats['total_projects']}")
        print()

        if stats["by_language"]:
            print(gum_style("By Language:", bold=True))
            for lang, data in stats["by_language"].items():
                if data["files"]:
                    avg_tokens = f"{data['avg_tokens']:.0f}" if data["avg_tokens"] else "N/A"
                    print(f"\n  {gum_style(lang, foreground='14')}:")
                    print(f"    Files:        {data['files']}")
                    print(f"    Chunks:       {data['chunks']}")
                    print(f"    ├─ file:        {data.get('files_chunks', 0)}")
                    print(f"    ├─ declaration: {data.get('declarations', 0)}")
                    print(f"    ├─ function:    {data['functions']}")
                    print(f"    ├─ class:       {data['classes']}")
                    print(f"    └─ method:      {data['methods']}")
                    print(f"    Avg tokens/chunk: {avg_tokens}")

        # Quality indicators
        if quality and quality.get("total_chunks", 0) > 0:
            callable_count = quality["callable_chunks"]
            sig_count = quality["with_signature"]
            sig_pct = (sig_count / callable_count * 100) if callable_count else 0
            file_pct = (quality["file_chunks"] / quality["total_chunks"] * 100)
            print()
            print(gum_style("Quality Indicators:", bold=True))
            print(f"  Signature coverage:    {sig_count}/{callable_count} function+method ({sig_pct:.1f}%)")
            print(f"  Tiny chunks (<30 tok): {quality['tiny_chunks']}")
            print(f"  Oversized (>600 tok):  {quality['oversized_chunks']}")
            print(f"  File chunk ratio:      {file_pct:.1f}%")

    finally:
        await db.close()


async def interactive_analyze():
    """Interactive analyze search quality logs."""
    from code_context.config import get_settings
    from scripts.analyze_logs import _load_entries, analyze

    log_path = get_settings().search_log_path
    if not log_path:
        print_warning("SEARCH_LOG_PATH not configured in .env")
        return

    entries = _load_entries(log_path)
    if not entries:
        print_warning(f"No log entries found at {log_path}")
        return

    options = [
        f"All ({len(entries)} queries)",
        "Last 20",
        "Last 50",
        "Last 100",
    ]
    choice = gum_choose(options, header="How many queries to analyze?")
    if not choice:
        return

    if "Last 20" in choice:
        entries = entries[-20:]
    elif "Last 50" in choice:
        entries = entries[-50:]
    elif "Last 100" in choice:
        entries = entries[-100:]

    analyze(entries)


async def interactive_benchmark():
    """Interactive benchmark — pick a project and run."""
    from scripts.benchmark_retrieval import BENCHMARK_QUERIES

    available = list(BENCHMARK_QUERIES.keys())
    if not available:
        print_warning("No benchmark queries defined.")
        return

    project_id = gum_choose(available, header="Select project to benchmark:")
    if not project_id:
        return

    save_choice = gum_choose(["Just run", "Run and save baseline"], header="Save baseline?")
    save_name = None
    if save_choice and "save" in save_choice.lower():
        save_name = gum_input("Baseline name (e.g. v1):")
        if not save_name:
            return

    await quick_benchmark(project_id, save_name, None)


async def main_menu():
    """Main interactive menu."""
    while True:
        print()

        # Check for active watchers to show count
        from cli.watcher import is_global_watcher_running, list_active_watchers
        active_watchers = list_active_watchers()
        global_running = is_global_watcher_running()
        n_individual = sum(1 for w in active_watchers if not w.get("is_global"))

        if global_running:
            watcher_label = "⚙️   Manage watchers (global + {n} individual)".format(n=n_individual)
        elif active_watchers:
            watcher_label = "⚙️   Manage watchers ({n} active)".format(n=n_individual)
        else:
            watcher_label = "⚙️   Manage watchers"

        global_label = "🌐  Global watcher (running)" if global_running else "🌐  Global watcher (start all projects)"

        menu_options = [
            "── Projects ──────────────────────",
            "   📋  List projects",
            "   📊  View project stats",
            "   ➕  Index new project",
            "── Sync ────────────────────────────",
            "   🔄  Sync project (reindex changes)",
            "   🔄  Sync all projects",
            "   🔍  Check sync status (dry-run)",
            "── Watchers ────────────────────────",
            f"   {global_label}",
            "   👁️   Start watcher (single project)",
            f"   {watcher_label}",
            "── Tools ───────────────────────────",
            "   📈  Global statistics",
            "   📊  Analyze search quality",
            "   🏁  Run benchmark",
            "   🧹  Prune orphaned chunks",
            "── Danger Zone ─────────────────────",
            "   🗑️   Delete project",
            "   💣  Reset database",
            "   ❌  Exit",
        ]

        choice = gum_choose(menu_options, header="What would you like to do?", height=20)

        if not choice or "Exit" in choice:
            print_info("Goodbye!")
            break

        # Skip separator selections
        if choice.startswith("──"):
            continue

        print()

        try:
            if "List projects" in choice:
                await list_projects()
            elif "View project stats" in choice:
                await view_project_stats()
            elif "Index new project" in choice:
                await index_project()
            elif "Check sync status" in choice:
                await check_sync_status()
            elif "Sync all projects" in choice:
                await quick_sync_all()
            elif "Sync project" in choice:
                await sync_project()
            elif "Global watcher" in choice:
                await watch_all_projects()
            elif "Start watcher" in choice:
                await watch_project()
            elif "Manage watchers" in choice:
                await manage_watchers()
            elif "Delete project" in choice:
                await delete_project()
            elif "Global statistics" in choice:
                await global_stats()
            elif "Analyze search quality" in choice:
                await interactive_analyze()
            elif "Run benchmark" in choice:
                await interactive_benchmark()
            elif "Prune orphaned" in choice:
                await prune_orphans()
            elif "Reset database" in choice:
                await reset_database()
        except KeyboardInterrupt:
            print()
            print_info("Interrupted.")
        except Exception as e:
            print_error(f"Error: {e}")

        # Pause before showing menu again
        pause()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Code Context manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick commands (non-interactive):
  --list                List all indexed projects
  --stats               Show global statistics
  --index PATH          Index a project (use --id to set custom ID)
  --id ID               Project ID for indexing (default: folder name)
  --delete ID           Delete a project by ID
  --check ID            Check if project is up to date (dry-run)
  --sync-all            Sync all indexed projects sequentially
  --watch-all           Start global watcher (all projects, single process)
  --stop-global         Stop the global watcher

Interactive mode (default):
  Run without arguments for interactive menu
        """,
    )
    parser.add_argument("--list", "-l", action="store_true", help="List all projects")
    parser.add_argument("--stats", "-s", action="store_true", help="Show global stats")
    parser.add_argument("--index", "-i", metavar="PATH", help="Index a project")
    parser.add_argument("--id", metavar="ID", help="Project ID (default: folder name)")
    parser.add_argument("--sync", metavar="ID", help="Sync project (reindex only changed files)")
    parser.add_argument("--check", metavar="ID", help="Check sync status (dry-run, no changes)")
    parser.add_argument("--delete", "-d", metavar="ID", help="Delete a project by ID")
    parser.add_argument("--force", "-f", action="store_true", help="Force reindex (with --index)")
    parser.add_argument("--watch", "-w", metavar="PATH", help="Start background watcher for project")
    parser.add_argument("--watchers", action="store_true", help="List active watchers")
    parser.add_argument("--stop-watcher", metavar="PATH", help="Stop watcher for project")
    parser.add_argument("--sync-all", action="store_true", help="Sync all projects sequentially")
    parser.add_argument("--stop-all-watchers", action="store_true", help="Stop all active watchers")
    parser.add_argument("--watch-all", action="store_true", help="Start global watcher for all projects")
    parser.add_argument("--stop-global", action="store_true", help="Stop the global watcher")
    parser.add_argument("--analyze", action="store_true", help="Analyze search quality logs")
    parser.add_argument("--benchmark", metavar="PROJECT", help="Run retrieval benchmark")
    parser.add_argument("--save-baseline", metavar="NAME", help="Save benchmark as baseline")
    parser.add_argument("--compare-baseline", metavar="NAME", help="Compare against baseline")

    args, extra_args = parser.parse_known_args()

    # Non-interactive quick commands
    if args.list:
        asyncio.run(list_projects())
        return
    if args.stats:
        asyncio.run(global_stats())
        return
    if args.index:
        asyncio.run(quick_index(args.index, project_id=args.id, force=args.force))
        return
    if args.sync_all:
        asyncio.run(quick_sync_all())
        return
    if args.sync:
        asyncio.run(quick_sync(args.sync))
        return
    if args.check:
        asyncio.run(quick_check(args.check))
        return
    if args.delete:
        asyncio.run(quick_delete(args.delete))
        return
    if args.watch:
        quick_start_watcher(args.watch)
        return
    if args.watchers:
        quick_list_watchers()
        return
    if args.stop_watcher:
        quick_stop_watcher(args.stop_watcher)
        return
    if args.stop_all_watchers:
        quick_stop_all_watchers()
        return
    if args.watch_all:
        quick_start_global_watcher()
        return
    if args.stop_global:
        quick_stop_global_watcher()
        return
    if args.analyze:
        quick_analyze(extra_args)
        return
    if args.benchmark:
        asyncio.run(quick_benchmark(args.benchmark, args.save_baseline, args.compare_baseline))
        return

    # Interactive mode requires gum
    if not gum_available():
        print("Error: gum is not installed (required for interactive mode).")
        print("Install it with: brew install gum")
        print("Or use quick commands: --list, --stats, --index, --delete")
        print("Run with --help for more info.")
        sys.exit(1)

    print_header()

    try:
        asyncio.run(main_menu())
    except KeyboardInterrupt:
        print()
        print_info("Goodbye!")


async def quick_index(path: str, project_id: str | None = None, force: bool = False):
    """Quick non-interactive index command."""
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    project_path = Path(path).expanduser().resolve()

    if not project_path.is_dir():
        print(f"Error: Not a directory: {project_path}")
        return

    # Default project_id to folder name if not specified
    if not project_id:
        project_id = project_path.name

    print(f"Indexing: {project_path}")
    print(f"Project ID: {project_id}")
    print("This may take a while for large projects...")

    db = DatabasePool()
    await db.initialize()

    try:
        voyage = VoyageClient()
        indexer = Indexer(db, voyage)

        stats = await indexer.index_project(
            project_root=str(project_path),
            project_id=project_id,
            force=force,
        )

        print(f"✓ Indexed {stats['indexed_files']} files as '{project_id}'")
        print(f"  Total files scanned: {stats['total_files']}")
        print(f"  Files skipped: {stats['skipped_files']}")
        print(f"  Total chunks: {stats['total_chunks']}")

        if stats["errors"]:
            print(f"  Errors: {len(stats['errors'])}")
    finally:
        await db.close()


async def quick_sync(project_id: str):
    """Quick non-interactive sync command."""
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    db = DatabasePool()
    await db.initialize()

    try:
        # Find the project
        projects = await db.list_projects()
        project = next((p for p in projects if p["project_id"] == project_id), None)

        if not project:
            print(f"Project not found: {project_id}")
            print("Available projects:")
            for p in projects:
                print(f"  - {p['project_id']}")
            return

        project_path = project["project_root"]
        print(f"Syncing: {project_path}")
        print("Checking for modified files...")

        voyage = VoyageClient()
        indexer = Indexer(db, voyage)

        stats = await indexer.index_project(
            project_root=project_path,
            project_id=project_id,
            force=False,  # Only changed files
        )

        deleted = stats.get("deleted_files", 0)
        if stats["indexed_files"] == 0 and deleted == 0:
            print("✓ Already up to date! No changes detected.")
        else:
            parts = []
            if stats["indexed_files"] > 0:
                parts.append(f"{stats['indexed_files']} reindexed")
            if deleted > 0:
                parts.append(f"{deleted} removed")
            print(f"✓ Synced: {', '.join(parts)}")

        print(f"  Files scanned:   {stats['total_files']}")
        print(f"  Files reindexed: {stats['indexed_files']}")
        print(f"  Files unchanged: {stats['skipped_files']}")
        if deleted > 0:
            print(f"  Files removed:   {deleted}")
        print(f"  Chunks created:  {stats['total_chunks']}")

        if stats["errors"]:
            print(f"  Errors: {len(stats['errors'])}")

    finally:
        await db.close()


async def quick_sync_all():
    """Sync all indexed projects sequentially."""
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    db = DatabasePool()
    await db.initialize()

    try:
        projects = await db.list_projects()

        if not projects:
            print("No projects indexed.")
            return

        print(f"Syncing {len(projects)} project(s)...\n")

        total_indexed = 0
        total_removed = 0
        total_errors = 0

        voyage = VoyageClient()
        indexer = Indexer(db, voyage)

        for i, project in enumerate(projects, 1):
            project_id = project["project_id"]
            project_path = project["project_root"]
            print(f"[{i}/{len(projects)}] {project_id} ({project_path})")

            if not Path(project_path).is_dir():
                print("  ⚠ Path not found, skipping")
                continue

            try:
                stats = await indexer.index_project(
                    project_root=project_path,
                    project_id=project_id,
                    force=False,
                    ensure_vector_index=False,
                )

                indexed = stats["indexed_files"]
                deleted = stats.get("deleted_files", 0)
                errors = len(stats.get("errors", []))

                total_indexed += indexed
                total_removed += deleted
                total_errors += errors

                if indexed == 0 and deleted == 0:
                    print("  ✓ Up to date")
                else:
                    parts = []
                    if indexed > 0:
                        parts.append(f"{indexed} indexed")
                    if deleted > 0:
                        parts.append(f"{deleted} removed")
                    print(f"  ✓ {', '.join(parts)}")
                if errors > 0:
                    print(f"  ⚠ {errors} error(s)")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                total_errors += 1

        if total_indexed > 0 or total_removed > 0:
            print("\nEnsuring vector index (single pass)...")
            await db.create_vector_index()

        print(f"\nDone: {total_indexed} indexed, {total_removed} removed, {total_errors} error(s)")

    finally:
        await db.close()


async def quick_check(project_id: str):
    """Quick non-interactive check sync status command."""
    from code_context.db.pool import DatabasePool
    from code_context.indexer import Indexer

    db = DatabasePool()
    await db.initialize()

    try:
        projects = await db.list_projects()
        project = next((p for p in projects if p["project_id"] == project_id), None)

        if not project:
            print(f"Project not found: {project_id}")
            print("Available projects:")
            for p in projects:
                print(f"  - {p['project_id']}")
            return

        project_path = project["project_root"]
        print(f"Checking: {project_path}")

        indexer = Indexer(db, None)
        status = await indexer.check_status(
            project_root=project_path,
            project_id=project_id,
        )

        print()
        if status["up_to_date"]:
            print(f"✓ Up to date ({status['total_indexed']} files)")
        else:
            new_count = len(status["new"])
            mod_count = len(status["modified"])
            del_count = len(status["deleted"])
            print(f"⚠ Out of sync: +{new_count} new, ~{mod_count} modified, -{del_count} deleted")
            if status["new"]:
                for f in status["new"][:5]:
                    print(f"  + {f}")
                if len(status["new"]) > 5:
                    print(f"  ... +{len(status['new']) - 5} more")
            if status["modified"]:
                for f in status["modified"][:5]:
                    print(f"  ~ {f}")
                if len(status["modified"]) > 5:
                    print(f"  ... +{len(status['modified']) - 5} more")
            if status["deleted"]:
                for f in status["deleted"][:5]:
                    print(f"  - {f}")
                if len(status["deleted"]) > 5:
                    print(f"  ... +{len(status['deleted']) - 5} more")
            if status.get("skipped_ineligible", 0) > 0:
                print(f"  ({status['skipped_ineligible']} files skipped - too large)")
            print(f"\nRun 'cc2 --sync {project_id}' to apply changes.")

    finally:
        await db.close()


async def quick_delete(project_id: str):
    """Quick non-interactive delete command."""
    from code_context.db.pool import DatabasePool

    db = DatabasePool()
    await db.initialize()

    try:
        result = await db.delete_project(project_id)
        if result["deleted_files"] > 0:
            print(f"✓ Deleted {result['deleted_files']} files and {result['deleted_chunks']} chunks")
        else:
            print(f"Project not found: {project_id}")
    finally:
        await db.close()


def quick_start_watcher(path: str):
    """Quick non-interactive start watcher command."""
    from cli.watcher import start_watcher_daemon

    project_path = Path(path).expanduser().resolve()

    if not project_path.is_dir():
        print(f"Error: Not a directory: {project_path}")
        return

    success, message = start_watcher_daemon(str(project_path))
    print(message)


def quick_list_watchers():
    """Quick non-interactive list watchers command."""
    from cli.watcher import list_active_watchers

    watchers = list_active_watchers()

    if not watchers:
        print("No active watchers")
        return

    print("Active watchers:")
    for w in watchers:
        started = w.get("started_at", "unknown")
        if isinstance(started, str) and "T" in started:
            started = started.split("T")[1].split(".")[0]
        if w.get("is_global"):
            print(f"  PID {w['pid']}: [GLOBAL] (all projects)")
        else:
            print(f"  PID {w['pid']}: {w['project_path']}")
        print(f"    Started: {started}")


def quick_stop_watcher(path: str):
    """Quick non-interactive stop watcher command."""
    from cli.watcher import stop_watcher

    project_path = str(Path(path).expanduser().resolve())

    if stop_watcher(project_path):
        print(f"✓ Stopped watcher for: {project_path}")
    else:
        print(f"No active watcher for: {project_path}")


def quick_start_global_watcher():
    """Quick non-interactive start global watcher command."""
    from cli.watcher import is_global_watcher_running, start_global_watcher_daemon

    if is_global_watcher_running():
        print("Global watcher is already running.")
        return

    success, message = start_global_watcher_daemon()
    print(message)


def quick_stop_global_watcher():
    """Quick non-interactive stop global watcher command."""
    from cli.watcher import stop_global_watcher

    if stop_global_watcher():
        print("✓ Global watcher stopped.")
    else:
        print("No global watcher running.")


def quick_stop_all_watchers():
    """Quick non-interactive stop all watchers command."""
    from cli.watcher import list_active_watchers, stop_watcher_by_pid

    watchers = list_active_watchers()

    if not watchers:
        print("No active watchers")
        return

    for w in watchers:
        stop_watcher_by_pid(w["pid"])
        label = "[GLOBAL]" if w.get("is_global") else w["project_path"]
        print(f"✓ Stopped watcher PID {w['pid']}: {label}")

    print(f"\n✓ Stopped {len(watchers)} watcher(s)")


def quick_analyze(extra_args: list[str]):
    """Quick non-interactive analyze command — delegates to scripts.analyze_logs."""
    from scripts.analyze_logs import main as analyze_main

    # Temporarily override sys.argv so argparse in analyze_logs works
    import sys
    original_argv = sys.argv
    sys.argv = ["analyze_logs"] + extra_args
    try:
        analyze_main()
    finally:
        sys.argv = original_argv


async def quick_benchmark(project_id: str, save_name: str | None, compare_name: str | None):
    """Quick non-interactive benchmark command."""
    from scripts.benchmark_retrieval import main_async
    await main_async(project_id, save_name, compare_name)


if __name__ == "__main__":
    main()
