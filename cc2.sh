#!/usr/bin/env bash
# code-context-v2 CLI wrapper
# Usage: cc2 [command] [options]
#
# Commands:
#   (none)          Interactive menu (requires gum)
#   list, -l        List all indexed projects
#   stats, -s       Show global statistics
#   index, -i       Index a project (requires path)
#   sync            Sync project (reindex only changed files)
#   check           Check if project needs syncing (dry-run)
#   delete, -d      Delete a project (requires id)
#   watch, -w       Start watcher for a project
#   watchers        List active watchers
#   sync-all        Sync all indexed projects
#   stop            Stop watcher for a project
#   stop-all        Stop all watchers
#
# Examples:
#   cc2                                    # Interactive menu
#   cc2 list                               # List projects
#   cc2 index ~/projects/myapp             # Index with auto-id
#   cc2 index ~/projects/myapp -id myapp   # Index with custom id
#   cc2 check myapp                        # Check if needs syncing
#   cc2 sync myapp                         # Sync project (only changes)
#   cc2 delete myapp                       # Delete project
#   cc2 watch ~/projects/myapp             # Start watcher

set -euo pipefail

BASE_DIR="${HOME}/.claude/mcp-servers/code-context-v2"
VENV_PYTHON="${BASE_DIR}/.venv/bin/python"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}→${NC} $1"; }
print_warn() { echo -e "${YELLOW}⚠${NC} $1"; }

check_deps() {
    if [[ ! -f "${VENV_PYTHON}" ]]; then
        print_error "Python venv not found at ${VENV_PYTHON}"
        print_info "Run: cd ${BASE_DIR} && uv venv && uv pip install -e ."
        exit 1
    fi
}

run_cli() {
    cd "${BASE_DIR}"
    "${VENV_PYTHON}" -m cli.manage "$@"
}

show_help() {
    echo -e "${PURPLE}code-context-v2 CLI${NC}"
    echo ""
    echo "Usage: cc2 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  (none)              Interactive menu (requires gum)"
    echo "  list, -l            List all indexed projects"
    echo "  stats, -s           Show global statistics"
    echo "  index PATH [-id ID] Index a project"
    echo "  sync ID             Sync project (reindex only changed files)"
    echo "  sync-all            Sync all indexed projects"
    echo "  check ID            Check sync status (dry-run, no changes)"
    echo "  delete ID           Delete a project by ID"
    echo "  watch PATH          Start background watcher"
    echo "  watchers            List active watchers"
    echo "  stop PATH           Stop watcher for project"
    echo "  stop-all            Stop all watchers"
    echo ""
    echo "Examples:"
    echo "  cc2                                  # Interactive menu"
    echo "  cc2 list                             # List projects"
    echo "  cc2 index ~/myproject                # Index (id = folder name)"
    echo "  cc2 index ~/myproject -id my-proj    # Index with custom id"
    echo "  cc2 check my-proj                    # Check if needs syncing"
    echo "  cc2 sync my-proj                     # Sync (only changed files)"
    echo "  cc2 sync-all                         # Sync all projects"
    echo "  cc2 delete my-proj                   # Delete project"
    echo "  cc2 watch ~/myproject                # Start watcher"
    echo ""
    echo "Pipeline: voyage-4-large (index) → voyage-4-lite (query) → rerank-2.5"
}

cmd_list() {
    run_cli --list
}

cmd_stats() {
    run_cli --stats
}

cmd_index() {
    local path=""
    local project_id=""
    local force=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -id|--id)
                project_id="$2"
                shift 2
                ;;
            -f|--force)
                force="--force"
                shift
                ;;
            *)
                if [[ -z "${path}" ]]; then
                    path="$1"
                fi
                shift
                ;;
        esac
    done

    if [[ -z "${path}" ]]; then
        print_error "Path required. Usage: cc2 index PATH [-id ID]"
        exit 1
    fi

    # Expand path
    path=$(realpath -m "${path/#\~/$HOME}")

    if [[ ! -d "${path}" ]]; then
        print_error "Not a directory: ${path}"
        exit 1
    fi

    local args=(--index "${path}")
    [[ -n "${project_id}" ]] && args+=(--id "${project_id}")
    [[ -n "${force}" ]] && args+=(--force)

    run_cli "${args[@]}"
}

cmd_sync() {
    local project_id="$1"

    if [[ -z "${project_id}" ]]; then
        print_error "Project ID required. Usage: cc2 sync ID"
        print_info "Use 'cc2 list' to see available projects"
        exit 1
    fi

    run_cli --sync "${project_id}"
}

cmd_sync_all() {
    run_cli --sync-all
}

cmd_check() {
    local project_id="$1"

    if [[ -z "${project_id}" ]]; then
        print_error "Project ID required. Usage: cc2 check ID"
        print_info "Use 'cc2 list' to see available projects"
        exit 1
    fi

    run_cli --check "${project_id}"
}

cmd_delete() {
    local project_id="$1"

    if [[ -z "${project_id}" ]]; then
        print_error "Project ID required. Usage: cc2 delete ID"
        exit 1
    fi

    run_cli --delete "${project_id}"
}

cmd_watch() {
    local path="$1"

    if [[ -z "${path}" ]]; then
        print_error "Path required. Usage: cc2 watch PATH"
        exit 1
    fi

    path=$(realpath -m "${path/#\~/$HOME}")
    run_cli --watch "${path}"
}

cmd_watchers() {
    run_cli --watchers
}

cmd_stop() {
    local path="$1"

    if [[ -z "${path}" ]]; then
        print_error "Path required. Usage: cc2 stop PATH"
        exit 1
    fi

    path=$(realpath -m "${path/#\~/$HOME}")
    run_cli --stop-watcher "${path}"
}

cmd_stop_all() {
    run_cli --stop-all-watchers
}

cmd_interactive() {
    if ! command -v gum >/dev/null 2>&1; then
        print_warn "gum not installed. Using text mode."
        print_info "Install gum for better experience: brew install gum"
    fi
    run_cli
}

main() {
    check_deps

    if [[ $# -eq 0 ]]; then
        cmd_interactive
        exit 0
    fi

    local cmd="$1"
    shift

    case "${cmd}" in
        help|--help|-h)
            show_help
            ;;
        list|--list|-l)
            cmd_list
            ;;
        stats|--stats|-s)
            cmd_stats
            ;;
        index|--index|-i)
            cmd_index "$@"
            ;;
        sync-all|--sync-all)
            cmd_sync_all
            ;;
        sync|--sync)
            cmd_sync "${1:-}"
            ;;
        check|--check)
            cmd_check "${1:-}"
            ;;
        delete|--delete|-d)
            cmd_delete "${1:-}"
            ;;
        watch|--watch|-w)
            cmd_watch "${1:-}"
            ;;
        watchers|--watchers)
            cmd_watchers
            ;;
        stop|--stop)
            cmd_stop "${1:-}"
            ;;
        stop-all|--stop-all)
            cmd_stop_all
            ;;
        *)
            print_error "Unknown command: ${cmd}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
