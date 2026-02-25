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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Directory for storing watcher state
WATCHERS_DIR = Path.home() / ".code-context" / "watchers"

# Global watcher constants
GLOBAL_WATCHER_SENTINEL = "__global__"
GLOBAL_WATCHER_PID_FILE = WATCHERS_DIR / "global.json"
GLOBAL_WATCHER_LOG = WATCHERS_DIR / "logs" / "global.log"


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


# ---------------------------------------------------------------------------
# Global watcher — single daemon for all indexed projects
# ---------------------------------------------------------------------------


def _compute_pool_max_size(n_projects: int) -> int:
    """Scale DB pool size based on project count."""
    return min(50, max(20, n_projects * 2 + 5))


def is_global_watcher_running() -> bool:
    """Check if the global watcher daemon is running."""
    if not GLOBAL_WATCHER_PID_FILE.exists():
        return False
    try:
        info = json.loads(GLOBAL_WATCHER_PID_FILE.read_text())
        pid = info.get("pid")
        if pid and is_process_running(pid):
            return True
        # Stale PID file — clean up
        GLOBAL_WATCHER_PID_FILE.unlink(missing_ok=True)
        return False
    except (json.JSONDecodeError, KeyError, OSError):
        GLOBAL_WATCHER_PID_FILE.unlink(missing_ok=True)
        return False


def stop_global_watcher() -> bool:
    """Stop the global watcher daemon."""
    if not GLOBAL_WATCHER_PID_FILE.exists():
        return False
    try:
        info = json.loads(GLOBAL_WATCHER_PID_FILE.read_text())
        pid = info.get("pid")
        if pid and is_process_running(pid):
            os.kill(pid, signal.SIGTERM)
            import time
            for _ in range(20):  # up to 2s grace
                if not is_process_running(pid):
                    break
                time.sleep(0.1)
            else:
                os.kill(pid, signal.SIGKILL)
        GLOBAL_WATCHER_PID_FILE.unlink(missing_ok=True)
        return True
    except (json.JSONDecodeError, OSError):
        GLOBAL_WATCHER_PID_FILE.unlink(missing_ok=True)
        return False


def get_global_watcher_log() -> str | None:
    """Get the global watcher log file path."""
    return str(GLOBAL_WATCHER_LOG) if GLOBAL_WATCHER_LOG.exists() else None


@dataclass
class GlobalWatcherState:
    """Shared state for the global watcher."""

    db: object  # DatabasePool
    indexer: object  # Indexer
    index_sem: asyncio.Semaphore
    known_files: dict[str, set[str]] = field(default_factory=dict)
    tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    shutdown: asyncio.Event = field(default_factory=asyncio.Event)


def _ts() -> str:
    """Short timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


async def _watch_single_project(
    state: GlobalWatcherState,
    project_id: str,
    project_root: str,
    poll_interval: int,
) -> None:
    """Poll loop for a single project inside the global watcher."""
    path = Path(project_root)
    tag = f"[{project_id}]"

    if not path.is_dir():
        print(f"[{_ts()}] {tag} Path not found: {path}", flush=True)
        return

    indexer = state.indexer

    # Initial file collection
    collected = indexer._collect_files(path)
    state.known_files[project_id] = {str(p) for p, _ in collected}
    print(f"[{_ts()}] {tag} Tracking {len(state.known_files[project_id])} files", flush=True)

    # Initial sync
    print(f"[{_ts()}] {tag} Initial sync...", flush=True)
    try:
        result = await indexer.index_project(str(path), project_id, force=False)
        indexed = result["indexed_files"]
        if indexed > 0:
            print(f"[{_ts()}] {tag} Synced {indexed} files ({result['total_chunks']} chunks)", flush=True)
        else:
            print(f"[{_ts()}] {tag} Up to date", flush=True)
    except Exception as e:
        print(f"[{_ts()}] {tag} Sync error: {e}", flush=True)

    last_poll_time = datetime.now().timestamp()

    try:
        while not state.shutdown.is_set():
            await asyncio.sleep(poll_interval)
            if state.shutdown.is_set():
                break

            poll_start = datetime.now().timestamp()
            known = state.known_files.get(project_id, set())

            # Scan for changes
            collected = indexer._collect_files(path)
            current_files: set[str] = set()
            files_to_check: list[str] = []

            for file_path, _lang in collected:
                filepath_str = str(file_path)
                current_files.add(filepath_str)
                try:
                    mtime = file_path.stat().st_mtime
                    if mtime >= last_poll_time - 1:
                        files_to_check.append(filepath_str)
                except OSError:
                    pass

            # Detect deleted files
            deleted_files = known - current_files
            for filepath in deleted_files:
                filename = Path(filepath).name
                print(f"[{_ts()}] {tag} Deleted: {filename}")
                await indexer.remove_file(filepath)

            # Update known files
            state.known_files[project_id] = current_files

            # Index changed files with shared semaphore
            indexed_count = 0
            if files_to_check and not state.shutdown.is_set():
                async def _index_one(fp: str):
                    async with state.index_sem:
                        return fp, await indexer.index_file(
                            fp, str(path), project_id, force=False
                        )

                results = await asyncio.gather(
                    *[_index_one(fp) for fp in files_to_check],
                    return_exceptions=True,
                )

                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        filename = Path(files_to_check[i]).name
                        print(f"[{_ts()}] {tag} Error: {filename}: {res}")
                    else:
                        fp, index_result = res
                        if index_result["indexed"]:
                            indexed_count += 1
                            filename = Path(fp).name
                            print(f"[{_ts()}] {tag} Indexed: {filename} ({index_result['chunks']} chunks)")

            last_poll_time = poll_start

            if indexed_count > 0 or deleted_files:
                print(f"[{_ts()}] {tag} Sync: {indexed_count} indexed, {len(deleted_files)} deleted", flush=True)

    except asyncio.CancelledError:
        print(f"[{_ts()}] {tag} Task cancelled", flush=True)


async def _project_refresh_loop(
    state: GlobalWatcherState,
    poll_interval: int,
    refresh_interval: int = 300,
) -> None:
    """Periodically refresh the project list to detect additions/removals."""
    try:
        while not state.shutdown.is_set():
            await asyncio.sleep(refresh_interval)
            if state.shutdown.is_set():
                break

            print(f"[{_ts()}] [global] Refreshing project list...", flush=True)

            projects = await state.db.list_projects()
            current_ids = {p["project_id"]: p["project_root"] for p in projects}
            tracked_ids = set(state.tasks.keys())

            # New projects
            for pid, proot in current_ids.items():
                if pid not in tracked_ids:
                    if not Path(proot).is_dir():
                        print(f"[{_ts()}] [global] Skipping {pid}: path not found", flush=True)
                        continue
                    print(f"[{_ts()}] [global] Adding project: {pid}", flush=True)
                    state.known_files[pid] = set()
                    task = asyncio.create_task(
                        _watch_single_project(state, pid, proot, poll_interval),
                        name=f"watch-{pid}",
                    )
                    state.tasks[pid] = task

            # Removed projects
            for pid in tracked_ids - set(current_ids.keys()):
                print(f"[{_ts()}] [global] Removing project: {pid}", flush=True)
                task = state.tasks.pop(pid, None)
                if task:
                    task.cancel()
                state.known_files.pop(pid, None)

            print(f"[{_ts()}] [global] Watching {len(state.tasks)} projects", flush=True)

    except asyncio.CancelledError:
        pass


async def run_global_watcher(poll_interval: int = 30, refresh_interval: int = 300):
    """Run the global watcher — single process for all indexed projects."""
    from code_context.db.pool import DatabasePool
    from code_context.embedding.voyage import VoyageClient
    from code_context.indexer import Indexer

    # Load initial project list with a temporary pool
    tmp_db = DatabasePool()
    await tmp_db.initialize()
    projects = await tmp_db.list_projects()
    await tmp_db.close()

    n = len(projects)
    print(f"[{_ts()}] [global] Found {n} indexed project(s)", flush=True)

    # Create shared resources with scaled pool
    pool_max = _compute_pool_max_size(n)
    print(f"[{_ts()}] [global] DB pool max_size={pool_max}", flush=True)

    db = DatabasePool(max_size=pool_max)
    await db.initialize()
    voyage = VoyageClient()
    indexer = Indexer(db, voyage)

    from code_context.config import get_settings
    settings = get_settings()

    state = GlobalWatcherState(
        db=db,
        indexer=indexer,
        index_sem=asyncio.Semaphore(settings.index_concurrency),
    )

    # Stagger startup: spread initial syncs across the poll interval
    stagger_delay = poll_interval / max(n, 1) if n > 0 else 0
    for i, p in enumerate(projects):
        pid = p["project_id"]
        proot = p["project_root"]
        if not Path(proot).is_dir():
            print(f"[{_ts()}] [global] Skipping {pid}: path not found", flush=True)
            continue
        state.known_files[pid] = set()
        task = asyncio.create_task(
            _watch_single_project(state, pid, proot, poll_interval),
            name=f"watch-{pid}",
        )
        state.tasks[pid] = task
        if stagger_delay > 0 and i < n - 1:
            await asyncio.sleep(stagger_delay)

    # Start refresh loop
    refresh_task = asyncio.create_task(
        _project_refresh_loop(state, poll_interval, refresh_interval),
        name="project-refresh",
    )

    # Signal handling
    def shutdown_handler(signum, frame):
        print(f"\n[{_ts()}] [global] Shutdown requested...", flush=True)
        state.shutdown.set()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Wait for shutdown
    await state.shutdown.wait()

    # Cancel all tasks
    print(f"[{_ts()}] [global] Cancelling {len(state.tasks)} project task(s)...", flush=True)
    refresh_task.cancel()
    for task in state.tasks.values():
        task.cancel()
    all_tasks = list(state.tasks.values()) + [refresh_task]
    await asyncio.gather(*all_tasks, return_exceptions=True)

    await db.close()
    GLOBAL_WATCHER_PID_FILE.unlink(missing_ok=True)
    print(f"[{_ts()}] [global] Shutdown complete.", flush=True)


def start_global_watcher_daemon(
    poll_interval: int = 30,
    refresh_interval: int = 300,
) -> tuple[bool, str]:
    """Start the global watcher daemon in the background.

    Returns (success, message).
    """
    if is_global_watcher_running():
        info = json.loads(GLOBAL_WATCHER_PID_FILE.read_text())
        return False, f"Global watcher already running (PID: {info['pid']})"

    pid = os.fork()

    if pid > 0:
        # Parent — wait up to 1.5s for daemon to write PID file
        import time
        for _ in range(15):
            if GLOBAL_WATCHER_PID_FILE.exists():
                info = json.loads(GLOBAL_WATCHER_PID_FILE.read_text())
                return True, f"Global watcher started (PID: {info['pid']})"
            time.sleep(0.1)
        return False, "Failed to start global watcher"

    else:
        # Child — daemonize
        os.setsid()
        pid2 = os.fork()
        if pid2 > 0:
            os._exit(0)

        # Redirect I/O to log
        log_dir = WATCHERS_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        sys.stdin.close()
        log_fd = open(GLOBAL_WATCHER_LOG, "a", buffering=1)
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
        sys.stdout = log_fd
        sys.stderr = log_fd

        # Save PID file
        ensure_watchers_dir()
        info = {
            "project_path": GLOBAL_WATCHER_SENTINEL,
            "pid": os.getpid(),
            "started_at": datetime.now().isoformat(),
            "is_global": True,
            "poll_interval": poll_interval,
            "refresh_interval": refresh_interval,
        }
        GLOBAL_WATCHER_PID_FILE.write_text(json.dumps(info, indent=2))

        print(f"\n{'='*60}")
        print(f"Global watcher started at {datetime.now().isoformat()}")
        print(f"PID: {os.getpid()}")
        print(f"Poll: {poll_interval}s | Refresh: {refresh_interval}s")
        print(f"{'='*60}\n")

        try:
            asyncio.run(run_global_watcher(poll_interval, refresh_interval))
        except Exception as e:
            print(f"Global watcher error: {e}")
        finally:
            GLOBAL_WATCHER_PID_FILE.unlink(missing_ok=True)

        os._exit(0)


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
