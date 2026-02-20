#!/usr/bin/env python3
"""Background watcher daemon for Code Context.

This module provides functionality to run file watchers in the background
and manage them (list, stop).

PID files are stored in ~/.code-context/watchers/
"""

import asyncio
import hashlib
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Directory for storing watcher state
WATCHERS_DIR = Path.home() / ".code-context" / "watchers"


def get_watcher_id(project_path: str) -> str:
    """Generate a unique ID for a watcher based on project path."""
    return hashlib.md5(project_path.encode()).hexdigest()[:12]


def get_pid_file(project_path: str) -> Path:
    """Get PID file path for a project."""
    watcher_id = get_watcher_id(project_path)
    return WATCHERS_DIR / f"{watcher_id}.json"


def ensure_watchers_dir():
    """Ensure watchers directory exists."""
    WATCHERS_DIR.mkdir(parents=True, exist_ok=True)


def save_watcher_info(project_path: str, pid: int):
    """Save watcher info to PID file."""
    ensure_watchers_dir()
    info = {
        "project_path": project_path,
        "pid": pid,
        "started_at": datetime.now().isoformat(),
    }
    pid_file = get_pid_file(project_path)
    pid_file.write_text(json.dumps(info, indent=2))


def remove_watcher_info(project_path: str):
    """Remove watcher PID file."""
    pid_file = get_pid_file(project_path)
    if pid_file.exists():
        pid_file.unlink()


def is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def list_active_watchers() -> list[dict]:
    """List all active watchers."""
    ensure_watchers_dir()
    active = []

    for pid_file in WATCHERS_DIR.glob("*.json"):
        try:
            info = json.loads(pid_file.read_text())
            pid = info.get("pid")

            if pid and is_process_running(pid):
                info["status"] = "running"
                info["pid_file"] = str(pid_file)
                active.append(info)
            else:
                # Clean up stale PID file
                pid_file.unlink()
        except (json.JSONDecodeError, KeyError):
            # Invalid PID file, remove it
            pid_file.unlink()

    return active


def stop_watcher(project_path: str) -> bool:
    """Stop a watcher for a specific project."""
    pid_file = get_pid_file(project_path)

    if not pid_file.exists():
        return False

    try:
        info = json.loads(pid_file.read_text())
        pid = info.get("pid")

        if pid and is_process_running(pid):
            os.kill(pid, signal.SIGTERM)
            # Wait a bit for graceful shutdown
            for _ in range(10):
                if not is_process_running(pid):
                    break
                import time
                time.sleep(0.1)
            else:
                # Force kill if still running
                os.kill(pid, signal.SIGKILL)

        pid_file.unlink()
        return True
    except (json.JSONDecodeError, OSError):
        if pid_file.exists():
            pid_file.unlink()
        return False


def stop_watcher_by_pid(pid: int) -> bool:
    """Stop a watcher by PID."""
    try:
        if is_process_running(pid):
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                if not is_process_running(pid):
                    break
                import time
                time.sleep(0.1)
            else:
                os.kill(pid, signal.SIGKILL)

        # Clean up PID file
        for pid_file in WATCHERS_DIR.glob("*.json"):
            try:
                info = json.loads(pid_file.read_text())
                if info.get("pid") == pid:
                    pid_file.unlink()
                    break
            except (json.JSONDecodeError, KeyError):
                pass

        return True
    except OSError:
        return False


async def run_watcher(project_path: str, poll_interval: int = 30):
    """Run the file watcher (called by daemon process).

    Uses polling-based approach for reliability:
    - Scans filesystem every poll_interval seconds
    - Uses BLAKE3 hash to detect changes (fast, no false positives)
    - Only calls Voyage API when files actually changed

    This is more reliable than inotify which can miss events
    during high-activity periods (e.g., Next.js builds).
    """
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    path = Path(project_path).resolve()
    if not path.is_dir():
        print(f"Error: Not a directory: {path}")
        sys.exit(1)

    db = DatabasePool()
    await db.initialize()
    voyage = VoyageClient()
    indexer = Indexer(db, voyage)

    # Get project_id from database (project must be indexed first)
    projects = await db.list_projects()
    project_id = None
    for p in projects:
        if p["project_root"] == str(path):
            project_id = p["project_id"]
            break

    if not project_id:
        print(f"Error: Project not indexed. Run: cc2 index {path}")
        print("Watcher requires the project to be indexed first.")
        sys.exit(1)

    print(f"Project ID: {project_id}")
    print(f"Poll interval: {poll_interval}s")
    print(f"Watching: {path}")
    print(f"PID: {os.getpid()}")

    # Track known files for deletion detection (use fast os.walk scan)
    collected = indexer._collect_files(path)
    known_files: set[str] = {str(p) for p, _ in collected}

    print(f"Tracking {len(known_files)} files", flush=True)

    # Initial sync to catch any changes before watcher started
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⟳ Initial sync...", flush=True)
    try:
        result = await indexer.index_project(str(path), project_id, force=False)
        if result["indexed_files"] > 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⟳ Synced {result['indexed_files']} files ({result['total_chunks']} chunks)", flush=True)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⟳ All files up to date", flush=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⟳ Sync error: {e}", flush=True)

    # Shutdown flag
    shutdown_requested = False
    last_poll_time = datetime.now().timestamp()

    def shutdown_handler(signum, frame):
        nonlocal shutdown_requested
        print("\nShutting down watcher...")
        shutdown_requested = True

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        while not shutdown_requested:
            await asyncio.sleep(poll_interval)
            if shutdown_requested:
                break

            poll_start = datetime.now().timestamp()

            # Scan for changes using fast os.walk with pruning
            collected = indexer._collect_files(path)
            current_files: set[str] = set()
            files_to_check: list[str] = []

            for file_path, _lang in collected:
                filepath_str = str(file_path)
                current_files.add(filepath_str)

                # Only check files modified since last poll (with 1s buffer)
                try:
                    mtime = file_path.stat().st_mtime
                    if mtime >= last_poll_time - 1:
                        files_to_check.append(filepath_str)
                except OSError:
                    pass  # File might have been deleted

            # Detect deleted files
            deleted_files = known_files - current_files
            for filepath in deleted_files:
                filename = Path(filepath).name
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Deleted: {filename}")
                await indexer.remove_file(filepath)

            # Update known files
            known_files.clear()
            known_files.update(current_files)

            # Index changed files concurrently (hash check happens inside index_file)
            indexed_count = 0

            if files_to_check and not shutdown_requested:
                sem = asyncio.Semaphore(indexer.settings.index_concurrency)

                async def _index_one(fp: str):
                    async with sem:
                        return fp, await indexer.index_file(
                            fp, str(path), project_id, force=False
                        )

                results = await asyncio.gather(
                    *[_index_one(fp) for fp in files_to_check],
                    return_exceptions=True,
                )

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        filename = Path(files_to_check[i]).name
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error indexing {filename}: {result}")
                    else:
                        fp, index_result = result
                        if index_result["indexed"]:
                            indexed_count += 1
                            filename = Path(fp).name
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Indexed: {filename} ({index_result['chunks']} chunks)")

            # Update last poll time
            last_poll_time = poll_start

            # Summary
            if indexed_count > 0 or deleted_files:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⟳ Sync: {indexed_count} indexed, {len(deleted_files)} deleted", flush=True)
            else:
                # Heartbeat - show we're alive (every poll)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ♥ Poll: {len(files_to_check)} files checked", flush=True)

    except Exception as e:
        print(f"Watcher error: {e}")
    finally:
        await db.close()
        remove_watcher_info(str(path))


def start_watcher_daemon(project_path: str) -> tuple[bool, str]:
    """Start a watcher daemon in the background.

    Returns (success, message)
    """
    project_path = str(Path(project_path).resolve())

    # Check if already watching
    watchers = list_active_watchers()
    for w in watchers:
        if w["project_path"] == project_path:
            return False, f"Already watching (PID: {w['pid']})"

    # Fork to create daemon
    pid = os.fork()

    if pid > 0:
        # Parent process - wait a moment and check if child started
        import time
        time.sleep(0.5)

        # Verify the watcher started
        pid_file = get_pid_file(project_path)
        if pid_file.exists():
            info = json.loads(pid_file.read_text())
            return True, f"Watcher started (PID: {info['pid']})"
        return False, "Failed to start watcher"

    else:
        # Child process - become daemon
        os.setsid()  # Create new session

        # Second fork to prevent zombie
        pid2 = os.fork()
        if pid2 > 0:
            os._exit(0)

        # Redirect stdout/stderr to log file
        log_dir = WATCHERS_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        watcher_id = get_watcher_id(project_path)
        log_file = log_dir / f"{watcher_id}.log"

        # Close standard file descriptors
        sys.stdin.close()

        # Redirect stdout and stderr (unbuffered)
        log_fd = open(log_file, "a", buffering=1)  # Line buffered
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
        # Also set unbuffered mode for Python
        sys.stdout = log_fd
        sys.stderr = log_fd

        # Save PID file
        save_watcher_info(project_path, os.getpid())

        print(f"\n{'='*60}")
        print(f"Watcher started at {datetime.now().isoformat()}")
        print(f"Project: {project_path}")
        print(f"PID: {os.getpid()}")
        print(f"{'='*60}\n")

        # Run the watcher
        try:
            asyncio.run(run_watcher(project_path))
        except Exception as e:
            print(f"Watcher error: {e}")
        finally:
            remove_watcher_info(project_path)

        os._exit(0)


def get_watcher_log(project_path: str) -> str | None:
    """Get the log file path for a watcher."""
    watcher_id = get_watcher_id(project_path)
    log_file = WATCHERS_DIR / "logs" / f"{watcher_id}.log"
    return str(log_file) if log_file.exists() else None


if __name__ == "__main__":
    # CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="Watcher daemon manager")
    parser.add_argument("--start", metavar="PATH", help="Start watcher for project")
    parser.add_argument("--stop", metavar="PATH", help="Stop watcher for project")
    parser.add_argument("--list", action="store_true", help="List active watchers")

    args = parser.parse_args()

    if args.list:
        watchers = list_active_watchers()
        if watchers:
            print("Active watchers:")
            for w in watchers:
                print(f"  PID {w['pid']}: {w['project_path']}")
                print(f"    Started: {w['started_at']}")
        else:
            print("No active watchers")

    elif args.start:
        success, msg = start_watcher_daemon(args.start)
        print(msg)
        sys.exit(0 if success else 1)

    elif args.stop:
        if stop_watcher(args.stop):
            print(f"Stopped watcher for: {args.stop}")
        else:
            print(f"No active watcher for: {args.stop}")

    else:
        parser.print_help()
