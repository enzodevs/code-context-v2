"""FastMCP Server - SOTA Context Engineering for Codebases.

Exposes exactly 2 tools to minimize context window usage:
1. search_codebase - Semantic search across entire indexed codebase
2. search_by_file - Search within a specific file (for local refactoring)
"""

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from pydantic import BaseModel

from code_context.config import get_settings
from code_context.db.pool import DatabasePool, get_db_pool
from code_context.embedding.voyage import VoyageClient, get_voyage_client
from code_context.retrieval.pipeline import RetrievalPipeline

# Configure logging
logging.basicConfig(
    level=getattr(logging, get_settings().log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Response models for structured output
class CodeChunkResponse(BaseModel):
    filepath: str
    symbol_name: str | None = None
    chunk_type: str
    start_line: int
    end_line: int
    relevance_score: float
    code: str


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[CodeChunkResponse]


# Shared app lifecycle
@asynccontextmanager
async def app_lifespan(_server: FastMCP):
    """Initialize shared resources once and cleanly close on shutdown."""
    db = await get_db_pool()
    get_voyage_client()
    logger.info("Shared database pool and Voyage client initialized")
    try:
        yield {}
    finally:
        await db.close()
        logger.info("Shared database pool closed")


# Dependency injection for shared database and voyage clients
async def get_database() -> DatabasePool:
    """Shared database connection pool dependency."""
    return await get_db_pool()


def get_voyage() -> VoyageClient:
    """Shared Voyage AI client dependency."""
    return get_voyage_client()


def get_pipeline(
    db: DatabasePool = Depends(get_database),
    voyage: VoyageClient = Depends(get_voyage),
) -> RetrievalPipeline:
    """Retrieval pipeline dependency."""
    return RetrievalPipeline(db, voyage)


# Create FastMCP server
mcp = FastMCP(
    name="code-context-v2",
    lifespan=app_lifespan,
    instructions="""SOTA Context Engineering MCP Server for Codebases and Literature.

CODE SEARCH:
1. Call list_projects to get project IDs
2. Use search_codebase(query, project) or search_by_file(filepath, query, project)

LITERATURE SEARCH:
1. Call list_books to get book IDs
2. Use search_literature(query, book?) - book param is optional""",
)


@mcp.tool
async def search_codebase(
    query: str,
    project: str,
    top_k: int = 8,
    language: Literal["typescript", "javascript", "python", "java"] | None = None,
    chunk_type: Literal["function", "class", "method", "module"] | None = None,
    file_type: Literal["code", "docs", "all"] | None = None,
    directory: str | None = None,
    search_intent: Literal["implementation", "definition", "usage", "debug"] | None = None,
    pipeline: RetrievalPipeline = Depends(get_pipeline),
) -> SearchResponse:
    """Semantic search across the indexed codebase.

    IMPORTANT: Use list_projects FIRST to get the project ID, then use it here.

    Args:
        query: Natural language description of what you're looking for.
               Examples: "authentication middleware", "database connection handling",
               "error handling in API routes", "user validation logic"
        project: Project ID from list_projects (e.g., "bingo-facil", "my-app").
                 This is REQUIRED. Use list_projects first to see available projects.
        top_k: Maximum number of results to return (default: 8, max: 20)
        language: Optional filter by programming language
        chunk_type: Optional filter by code structure type
        file_type: "code" for code files only (ts, js, py, java, go, rust, sql),
                   "docs" for documentation files only (markdown),
                   "all" or None for all file types including docs and config
        directory: Scope search to a specific directory within the project.
                   Uses path prefix matching. Examples: "src/components", "apps/api/src", "lib/utils"
        search_intent: Guides reranking toward a specific result type.
                   "implementation" - concrete code that builds the feature
                   "definition" - types, interfaces, schemas, data structures
                   "usage" - call sites and integration examples
                   "debug" - error handling, logging, edge cases

    Returns:
        SearchResponse with matching code chunks and pre-formatted context.
    """
    top_k = min(top_k, 20)  # Cap at 20

    logger.info(f"search_codebase: query='{query[:50]}...', project={project}, top_k={top_k}, file_type={file_type}, directory={directory}, intent={search_intent}")

    results = await pipeline.search(
        query=query,
        top_k=top_k,
        language=language,
        chunk_type=chunk_type,
        project=project,
        file_type=file_type,
        directory=directory,
        search_intent=search_intent,
    )

    chunks = [
        CodeChunkResponse(
            filepath=r.filepath,
            symbol_name=r.symbol_name,
            chunk_type=r.chunk_type,
            start_line=r.start_line,
            end_line=r.end_line,
            relevance_score=r.relevance_score,
            code=r.chunk_text,
        )
        for r in results
    ]

    return SearchResponse(
        query=query,
        total_results=len(chunks),
        results=chunks,
    )


@mcp.tool
async def search_by_file(
    filepath: str,
    query: str,
    project: str,
    top_k: int = 5,
    search_intent: Literal["implementation", "definition", "usage", "debug"] | None = None,
    pipeline: RetrievalPipeline = Depends(get_pipeline),
) -> SearchResponse:
    """Search within a specific file - useful for local refactoring.

    When you need to understand or modify a specific file, use this tool
    to find relevant sections within that file.

    Args:
        filepath: Full path to the file to search within.
                  Must be an indexed file in the codebase.
        query: What you're looking for within the file.
               Examples: "error handling", "validation logic", "main function"
        project: Project ID from list_projects (REQUIRED).
        top_k: Maximum number of results (default: 5, max: 10)
        search_intent: Guides reranking toward a specific result type.
                   "implementation" - concrete code that builds the feature
                   "definition" - types, interfaces, schemas, data structures
                   "usage" - call sites and integration examples
                   "debug" - error handling, logging, edge cases

    Returns:
        SearchResponse with matching code chunks from the specified file.
    """
    top_k = min(top_k, 10)  # Cap at 10 for file-scoped search

    logger.info(f"search_by_file: filepath='{filepath}', query='{query[:50]}...', project={project}, intent={search_intent}")

    results = await pipeline.search_file(
        filepath=filepath,
        query=query,
        top_k=top_k,
        project=project,
        search_intent=search_intent,
    )

    chunks = [
        CodeChunkResponse(
            filepath=r.filepath,
            symbol_name=r.symbol_name,
            chunk_type=r.chunk_type,
            start_line=r.start_line,
            end_line=r.end_line,
            relevance_score=r.relevance_score,
            code=r.chunk_text,
        )
        for r in results
    ]

    return SearchResponse(
        query=query,
        total_results=len(chunks),
        results=chunks,
    )


class ProjectInfo(BaseModel):
    id: str
    path: str
    file_count: int
    chunk_count: int
    languages: list[str]
    total_loc: int


class ProjectListResponse(BaseModel):
    total_projects: int
    projects: list[ProjectInfo]


@mcp.tool
async def list_projects(
    db: DatabasePool = Depends(get_database),
) -> ProjectListResponse:
    """List all indexed projects available for search.

    Use this tool FIRST to discover which projects are indexed.
    The returned 'id' field is the short project identifier to use
    in search_codebase and search_by_file.

    Returns:
        ProjectListResponse with list of all indexed projects and their stats.
    """
    logger.info("list_projects: fetching all indexed projects")

    projects = await db.list_projects()

    project_infos = [
        ProjectInfo(
            id=p["project_id"],
            path=p["project_root"],
            file_count=p["file_count"],
            chunk_count=p["chunk_count"],
            languages=[lang for lang in p["languages"] if lang],
            total_loc=p["total_loc"] or 0,
        )
        for p in projects
    ]

    return ProjectListResponse(
        total_projects=len(project_infos),
        projects=project_infos,
    )


class BookInfo(BaseModel):
    id: str
    title: str
    author: str
    category: str


class BookListResponse(BaseModel):
    total: int
    books: list[BookInfo]


class BookChunkResponse(BaseModel):
    book_id: str
    book_title: str
    chapter: str
    content: str
    relevance: float


class LiteratureSearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[BookChunkResponse]


@mcp.tool
async def list_books(
    db: DatabasePool = Depends(get_database),
) -> BookListResponse:
    """List indexed books. Returns book IDs for use in search_literature."""
    logger.info("list_books: fetching indexed books")

    books = await db.list_books()

    return BookListResponse(
        total=len(books),
        books=[
            BookInfo(id=b["id"], title=b["title"], author=b["author"], category=b["category"])
            for b in books
        ],
    )


@mcp.tool
async def search_literature(
    query: str,
    book: str | None = None,
    category: str | None = None,
    top_k: int = 5,
    pipeline: RetrievalPipeline = Depends(get_pipeline),
) -> LiteratureSearchResponse:
    """Search software literature. Use list_books first to get book IDs."""
    top_k = min(top_k, 8)

    logger.info(f"search_literature: query='{query[:50]}...', book={book}, category={category}, top_k={top_k}")

    # Get query embedding
    query_embedding = await pipeline.voyage.embed_query(query)

    # Vector search with more candidates for reranking
    candidates = await pipeline.db.search_book_chunks(
        query_embedding=query_embedding,
        book_id=book,
        limit=25,
        category=category,
    )

    if not candidates:
        return LiteratureSearchResponse(query=query, total_results=0, results=[])

    # Rerank for precision
    docs = [c["content"] for c in candidates]
    reranked = await pipeline.voyage.rerank(query, docs, top_k=top_k)

    results = []
    for idx, score in reranked:
        c = candidates[idx]
        results.append(
            BookChunkResponse(
                book_id=c["book_id"],
                book_title=c["book_title"],
                chapter=c["chapter"],
                content=c["content"],
                relevance=score,
            )
        )

    return LiteratureSearchResponse(
        query=query,
        total_results=len(results),
        results=results,
    )


def main():
    """Entry point for the MCP server."""
    logger.info("Starting Code Context MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
