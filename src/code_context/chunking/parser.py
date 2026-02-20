"""AST-based hierarchical code parser using tree-sitter for multiple languages.

Implements 3-level hierarchical chunking:
1. file-level: Complete file for broad context
2. declaration-level: Top-level exports/declarations (const, class, interface)
3. function-level: Individual functions/methods for precise retrieval

Supported languages:
- TypeScript/JavaScript (.ts, .tsx, .js, .jsx, .mjs, .cjs)
- Python (.py, .pyi)
- Java (.java)
- Go (.go)
- Rust (.rs)
- SQL (.sql)
- Markdown (.md, .mdx)
- JSON (.json)
- YAML (.yml, .yaml)
- TOML (.toml)
- CSS (.css, .scss, .less)
- HTML (.html, .htm)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import blake3
import tiktoken
import tree_sitter_css as ts_css
import tree_sitter_go as ts_go
import tree_sitter_html as ts_html
import tree_sitter_java as ts_java
import tree_sitter_javascript as ts_javascript
import tree_sitter_json as ts_json
import tree_sitter_markdown as ts_markdown
import tree_sitter_python as ts_python
import tree_sitter_rust as ts_rust
import tree_sitter_sql as ts_sql
import tree_sitter_toml as ts_toml
import tree_sitter_typescript as ts_typescript
import tree_sitter_yaml as ts_yaml
from tree_sitter import Language, Parser, Node

from code_context.config import get_settings

logger = logging.getLogger(__name__)

# Token encoder for counting
_encoder = tiktoken.get_encoding("cl100k_base")


@dataclass
class ParsedChunk:
    """A parsed code chunk with metadata."""

    text: str
    start_line: int
    end_line: int
    chunk_type: str  # file | declaration | function | class | method | section | statement
    symbol_name: str | None = None
    context: dict = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = len(_encoder.encode(self.text))


# Language configuration
LANGUAGE_CONFIG = {
    "typescript": {
        "extensions": {".ts", ".tsx"},
        "language": Language(ts_typescript.language_typescript()),
        "node_types": {
            "function": {
                "function_declaration",
                "arrow_function",
                "function_expression",
                "generator_function_declaration",
            },
            "class": {"class_declaration"},
            "method": {"method_definition", "public_field_definition"},
            "import": {"import_statement"},
            "declaration": {
                "export_statement",
                "lexical_declaration",
                "variable_declaration",
                "class_declaration",
                "function_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
            },
        },
        "name_field": "name",
    },
    "javascript": {
        "extensions": {".js", ".jsx", ".mjs", ".cjs"},
        "language": Language(ts_javascript.language()),
        "node_types": {
            "function": {
                "function_declaration",
                "arrow_function",
                "function_expression",
                "generator_function_declaration",
            },
            "class": {"class_declaration"},
            "method": {"method_definition"},
            "import": {"import_statement"},
            "declaration": {
                "export_statement",
                "lexical_declaration",
                "variable_declaration",
                "class_declaration",
                "function_declaration",
            },
        },
        "name_field": "name",
    },
    "python": {
        "extensions": {".py", ".pyi"},
        "language": Language(ts_python.language()),
        "node_types": {
            "function": {"function_definition"},
            "class": {"class_definition"},
            "method": {"function_definition"},
            "import": {"import_statement", "import_from_statement"},
            "declaration": {
                "function_definition",
                "class_definition",
                "assignment",
                "decorated_definition",
            },
        },
        "name_field": "name",
    },
    "java": {
        "extensions": {".java"},
        "language": Language(ts_java.language()),
        "node_types": {
            "function": {"method_declaration", "constructor_declaration"},
            "class": {"class_declaration", "interface_declaration", "enum_declaration"},
            "method": {"method_declaration"},
            "import": {"import_declaration", "package_declaration"},
            "declaration": {
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            },
        },
        "name_field": "name",
    },
    "go": {
        "extensions": {".go"},
        "language": Language(ts_go.language()),
        "node_types": {
            "function": {"function_declaration", "method_declaration"},
            "class": {"type_declaration"},
            "method": {"method_declaration"},
            "import": {"import_declaration"},
            "declaration": {
                "function_declaration",
                "method_declaration",
                "type_declaration",
                "var_declaration",
                "const_declaration",
            },
        },
        "name_field": "name",
    },
    "rust": {
        "extensions": {".rs"},
        "language": Language(ts_rust.language()),
        "node_types": {
            "function": {"function_item"},
            "class": {"struct_item", "enum_item", "impl_item", "trait_item"},
            "method": {"function_item"},  # methods inside impl blocks
            "import": {"use_declaration"},
            "declaration": {
                "function_item",
                "struct_item",
                "enum_item",
                "impl_item",
                "trait_item",
                "const_item",
                "static_item",
                "type_item",
                "mod_item",
            },
        },
        "name_field": "name",
    },
    "sql": {
        "extensions": {".sql"},
        "language": Language(ts_sql.language()),
        "node_types": {
            "statement": {
                "statement",  # Container for all statements
                "create_table",
                "create_index",
                "create_view",
                "create_function",
                "create_procedure",
                "create_trigger",
                "alter_table",
                "select",
                "insert",
                "update",
                "delete",
            },
            "declaration": {
                "create_table",
                "create_index",
                "create_view",
                "create_function",
                "create_procedure",
            },
        },
        "name_field": "name",
        "chunk_strategy": "statement",
    },
    "markdown": {
        "extensions": {".md", ".mdx"},
        "language": Language(ts_markdown.language()),
        "node_types": {
            "section": {"section", "atx_heading"},
            "block": {"paragraph", "code_block", "fenced_code_block", "list"},
        },
        "name_field": None,
        "chunk_strategy": "section",
    },
    "json": {
        "extensions": {".json"},
        "language": Language(ts_json.language()),
        "node_types": {},
        "name_field": None,
        "chunk_strategy": "file_only",
    },
    "yaml": {
        "extensions": {".yml", ".yaml"},
        "language": Language(ts_yaml.language()),
        "node_types": {
            "block": {"block_mapping", "block_sequence"},
        },
        "name_field": None,
        "chunk_strategy": "file_only",
    },
    "toml": {
        "extensions": {".toml"},
        "language": Language(ts_toml.language()),
        "node_types": {
            "section": {"table", "table_array_element"},
        },
        "name_field": None,
        "chunk_strategy": "file_only",
    },
    "css": {
        "extensions": {".css", ".scss", ".less"},
        "language": Language(ts_css.language()),
        "node_types": {
            "rule": {"rule_set", "media_statement", "keyframes_statement"},
        },
        "name_field": None,
        "chunk_strategy": "file_only",
    },
    "html": {
        "extensions": {".html", ".htm"},
        "language": Language(ts_html.language()),
        "node_types": {
            "element": {"element", "script_element", "style_element"},
        },
        "name_field": None,
        "chunk_strategy": "file_only",
    },
}


def detect_language(filepath: str) -> str | None:
    """Detect language from file extension."""
    ext = Path(filepath).suffix.lower()
    for lang, config in LANGUAGE_CONFIG.items():
        if ext in config["extensions"]:
            return lang
    return None


class CodeParser:
    """Multi-language hierarchical code parser using tree-sitter.

    Extracts chunks at multiple levels depending on language:
    - Code files: file → declaration → function/method
    - SQL: file → statement
    - Markdown: file → section
    - Config files: file only
    """

    def __init__(self):
        self.settings = get_settings()
        self._parsers: dict[str, Parser] = {}

    def _get_parser(self, language: str) -> Parser:
        """Get or create parser for language."""
        if language not in self._parsers:
            if language not in LANGUAGE_CONFIG:
                raise ValueError(f"Unsupported language: {language}")

            parser = Parser()
            parser.language = LANGUAGE_CONFIG[language]["language"]
            self._parsers[language] = parser

        return self._parsers[language]

    def parse_file(self, filepath: str, content: str | None = None) -> list[ParsedChunk]:
        """Parse a file into hierarchical semantic chunks.

        Args:
            filepath: Path to the file
            content: Optional file content (reads from disk if not provided)

        Returns:
            List of ParsedChunk objects at multiple granularity levels
        """
        language = detect_language(filepath)
        if not language:
            logger.warning(f"Unsupported file type: {filepath}")
            return []

        if content is None:
            content = Path(filepath).read_text(encoding="utf-8")

        parser = self._get_parser(language)
        tree = parser.parse(content.encode("utf-8"))

        config = LANGUAGE_CONFIG[language]
        strategy = config.get("chunk_strategy", "code")

        chunks: list[ParsedChunk] = []

        # Collect imports for context (if applicable)
        imports = []
        if "import" in config.get("node_types", {}):
            imports = self._collect_imports(
                tree.root_node, content, config["node_types"].get("import", set())
            )

        # Check if this is a "small file" that shouldn't be fragmented
        line_count = content.count("\n") + 1
        is_small_file = line_count <= self.settings.small_file_lines

        # Level 1: File-level chunk (complete file)
        file_chunk = self._create_file_chunk(filepath, content, imports)
        if file_chunk:
            chunks.append(file_chunk)

        # For small files, ONLY return the file-level chunk (no fragmentation)
        if is_small_file and file_chunk:
            logger.debug(f"Small file ({line_count} lines), keeping as single chunk: {filepath}")
            return [file_chunk]  # Return early, no splitting

        # Strategy-specific chunking (only for larger files)
        if strategy == "code":
            # Level 2: Declaration-level chunks
            declaration_chunks = self._extract_declarations(
                tree.root_node, content, language, filepath, imports
            )
            chunks.extend(declaration_chunks)

            # Level 3: Function/method-level chunks
            function_chunks = self._extract_functions(
                tree.root_node, content, language, filepath, imports
            )
            chunks.extend(function_chunks)

        elif strategy == "statement":
            # SQL: Extract statements
            statement_chunks = self._extract_statements(
                tree.root_node, content, language, filepath
            )
            chunks.extend(statement_chunks)

        elif strategy == "section":
            # Markdown: Extract sections
            section_chunks = self._extract_sections(
                tree.root_node, content, filepath
            )
            chunks.extend(section_chunks)

        # file_only strategy: only file-level chunk (already added)

        # Filter by token count
        min_tokens = self.settings.chunk_min_tokens
        max_tokens = self.settings.chunk_max_tokens

        filtered = []
        for chunk in chunks:
            if chunk.token_count < min_tokens:
                continue
            if chunk.token_count > max_tokens:
                filtered.extend(self._split_large_chunk(chunk, max_tokens))
            else:
                filtered.append(chunk)

        # Deduplicate identical chunks (same text, different levels)
        seen_texts = set()
        deduplicated = []
        for chunk in filtered:
            text_hash = blake3.blake3(chunk.text.encode()).hexdigest()[:16]
            if text_hash not in seen_texts:
                deduplicated.append(chunk)
                seen_texts.add(text_hash)

        return deduplicated

    def _create_file_chunk(
        self, filepath: str, content: str, imports: list[str]
    ) -> ParsedChunk | None:
        """Create a file-level chunk for the complete file."""
        if not content.strip():
            return None

        token_count = len(_encoder.encode(content))

        # Skip if file is too small
        if token_count < 50:
            return None

        # Skip file-level chunk for very large files (will be split into smaller chunks)
        # For file_only languages (JSON, YAML, etc.), this means the file is skipped entirely
        max_file_tokens = 10000  # ~40KB of code
        if token_count > max_file_tokens:
            logger.info(f"Skipping file-level chunk ({token_count} tokens > {max_file_tokens}): {filepath}")
            return None

        return ParsedChunk(
            text=content,
            start_line=1,
            end_line=content.count("\n") + 1,
            chunk_type="file",
            symbol_name=Path(filepath).name,
            context={"filepath": filepath, "imports": imports[:5]},
            token_count=token_count,
        )

    def _extract_declarations(
        self,
        root: Node,
        source: str,
        language: str,
        filepath: str,
        imports: list[str],
    ) -> list[ParsedChunk]:
        """Extract declaration-level chunks (top-level exports/declarations)."""
        config = LANGUAGE_CONFIG[language]
        declaration_types = config["node_types"].get("declaration", set())
        import_types = config["node_types"].get("import", set())
        chunks: list[ParsedChunk] = []

        for child in root.children:
            if child.type in import_types:
                continue

            if child.type not in declaration_types:
                continue

            name = self._get_declaration_name(child, config.get("name_field"), source)
            text = source[child.start_byte:child.end_byte]

            if not text.strip():
                continue

            start_line = child.start_point[0] + 1
            end_line = child.end_point[0] + 1

            chunks.append(ParsedChunk(
                text=text,
                start_line=start_line,
                end_line=end_line,
                chunk_type="declaration",
                symbol_name=name,
                context={"filepath": filepath, "imports": imports[:5]},
            ))

        return chunks

    def _get_declaration_name(self, node: Node, name_field: str | None, source: str) -> str | None:
        """Extract name from a declaration node."""
        if not name_field:
            return None

        # For export statements, look inside
        if node.type == "export_statement":
            for child in node.children:
                if child.type in {"lexical_declaration", "variable_declaration"}:
                    for subchild in child.children:
                        if subchild.type == "variable_declarator":
                            name_node = subchild.child_by_field_name("name")
                            if name_node:
                                return name_node.text.decode("utf-8")
                elif child.type in {"function_declaration", "class_declaration", "interface_declaration", "type_alias_declaration"}:
                    name_node = child.child_by_field_name(name_field)
                    if name_node:
                        return name_node.text.decode("utf-8")
            for child in node.children:
                name = self._get_node_name(child, name_field)
                if name:
                    return name

        # Direct declaration
        name_node = node.child_by_field_name(name_field)
        if name_node:
            return name_node.text.decode("utf-8") if name_node.text else None

        # Fallback for lexical declarations
        if node.type in {"lexical_declaration", "variable_declaration"}:
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        return name_node.text.decode("utf-8")

        return self._get_node_name(node, name_field)

    def _extract_functions(
        self,
        root: Node,
        source: str,
        language: str,
        filepath: str,
        imports: list[str],
    ) -> list[ParsedChunk]:
        """Extract function/method-level chunks (fine-grained)."""
        config = LANGUAGE_CONFIG[language]
        node_types = config["node_types"]
        name_field = config.get("name_field")
        chunks: list[ParsedChunk] = []

        def visit(node: Node, parent_class: str | None = None):
            node_type = node.type

            # Handle export_statement
            if node_type == "export_statement":
                for child in node.children:
                    if child.type in node_types.get("class", set()):
                        class_name = self._get_node_name(child, name_field)
                        for grandchild in child.children:
                            visit(grandchild, class_name)
                        return

                    if child.type in node_types.get("function", set()):
                        func_name = self._get_node_name(child, name_field)
                        chunk = self._node_to_chunk(
                            child, source, "function", func_name, filepath, imports
                        )
                        if chunk:
                            chunks.append(chunk)
                        return

                for child in node.children:
                    visit(child, parent_class)
                return

            # Class
            if node_type in node_types.get("class", set()):
                class_name = self._get_node_name(node, name_field)
                for child in node.children:
                    visit(child, class_name)
                return

            # Method inside class
            if parent_class and node_type in node_types.get("method", set()):
                method_name = self._get_node_name(node, name_field)
                full_name = f"{parent_class}.{method_name}" if method_name else None
                chunk = self._node_to_chunk(
                    node, source, "method", full_name, filepath, imports, parent_class
                )
                if chunk:
                    chunks.append(chunk)
                return

            # Standalone function
            if not parent_class and node_type in node_types.get("function", set()):
                func_name = self._get_node_name(node, name_field)
                chunk = self._node_to_chunk(
                    node, source, "function", func_name, filepath, imports
                )
                if chunk:
                    chunks.append(chunk)
                return

            # Recurse
            for child in node.children:
                visit(child, parent_class)

        visit(root)
        return chunks

    def _extract_statements(
        self,
        root: Node,
        source: str,
        language: str,
        filepath: str,
    ) -> list[ParsedChunk]:
        """Extract SQL statements as chunks."""
        config = LANGUAGE_CONFIG[language]
        statement_types = config["node_types"].get("statement", set())
        chunks: list[ParsedChunk] = []

        def visit(node: Node):
            # For SQL, 'statement' is a container - look at children
            if node.type == "statement":
                for child in node.children:
                    if child.type in statement_types:
                        text = source[child.start_byte:child.end_byte]
                        if text.strip():
                            name = self._extract_sql_name(child, source)
                            chunks.append(ParsedChunk(
                                text=text,
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                chunk_type="statement",
                                symbol_name=name,
                                context={"filepath": filepath},
                            ))
                return

            for child in node.children:
                visit(child)

        visit(root)
        return chunks

    def _extract_sql_name(self, node: Node, source: str) -> str | None:
        """Extract name from SQL statement (table name, etc.)."""
        # Look for identifier after CREATE TABLE, CREATE INDEX, etc.
        for child in node.children:
            if child.type in {"identifier", "table_reference", "object_reference"}:
                return source[child.start_byte:child.end_byte]
            # Recurse one level for nested identifiers
            for subchild in child.children:
                if subchild.type == "identifier":
                    return source[subchild.start_byte:subchild.end_byte]
        return None

    def _extract_sections(
        self,
        root: Node,
        source: str,
        filepath: str,
    ) -> list[ParsedChunk]:
        """Extract markdown sections as chunks."""
        chunks: list[ParsedChunk] = []
        lines = source.split("\n")

        # Simple approach: split by headings
        current_section = []
        current_heading = None
        current_start = 1

        for i, line in enumerate(lines, 1):
            if line.startswith("#"):
                # Save previous section
                if current_section:
                    text = "\n".join(current_section)
                    if text.strip():
                        chunks.append(ParsedChunk(
                            text=text,
                            start_line=current_start,
                            end_line=i - 1,
                            chunk_type="section",
                            symbol_name=current_heading,
                            context={"filepath": filepath},
                        ))
                # Start new section
                current_section = [line]
                current_heading = line.lstrip("#").strip()
                current_start = i
            else:
                current_section.append(line)

        # Don't forget last section
        if current_section:
            text = "\n".join(current_section)
            if text.strip():
                chunks.append(ParsedChunk(
                    text=text,
                    start_line=current_start,
                    end_line=len(lines),
                    chunk_type="section",
                    symbol_name=current_heading,
                    context={"filepath": filepath},
                ))

        return chunks

    def _collect_imports(
        self, root: Node, source: str, import_types: set[str]
    ) -> list[str]:
        """Collect import statements from the file."""
        imports = []

        def visit(node: Node):
            if node.type in import_types:
                text = source[node.start_byte : node.end_byte].strip()
                imports.append(text)
            for child in node.children:
                visit(child)

        visit(root)
        return imports

    def _get_node_name(self, node: Node, name_field: str | None) -> str | None:
        """Extract the name from a node."""
        if not name_field:
            return None

        name_node = node.child_by_field_name(name_field)
        if name_node:
            return name_node.text.decode("utf-8") if name_node.text else None

        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8") if child.text else None

        return None

    def _node_to_chunk(
        self,
        node: Node,
        source: str,
        chunk_type: str,
        symbol_name: str | None,
        filepath: str,
        imports: list[str],
        parent_class: str | None = None,
    ) -> ParsedChunk | None:
        """Convert a tree-sitter node to a ParsedChunk."""
        text = source[node.start_byte : node.end_byte]
        if not text.strip():
            return None

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        context = {
            "filepath": filepath,
            "imports": imports[:5],
        }
        if parent_class:
            context["parent_class"] = parent_class

        return ParsedChunk(
            text=text,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            symbol_name=symbol_name,
            context=context,
        )

    def _split_large_chunk(
        self, chunk: ParsedChunk, max_tokens: int
    ) -> list[ParsedChunk]:
        """Split a large chunk into smaller pieces."""
        lines = chunk.text.split("\n")
        result = []
        current_lines = []
        current_tokens = 0

        for i, line in enumerate(lines):
            line_tokens = len(_encoder.encode(line))

            if current_tokens + line_tokens > max_tokens and current_lines:
                text = "\n".join(current_lines)
                result.append(
                    ParsedChunk(
                        text=text,
                        start_line=chunk.start_line + len(result) * len(current_lines),
                        end_line=chunk.start_line
                        + len(result) * len(current_lines)
                        + len(current_lines)
                        - 1,
                        chunk_type=chunk.chunk_type,
                        symbol_name=f"{chunk.symbol_name}_part{len(result) + 1}"
                        if chunk.symbol_name
                        else None,
                        context=chunk.context,
                    )
                )
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            text = "\n".join(current_lines)
            result.append(
                ParsedChunk(
                    text=text,
                    start_line=chunk.start_line + len(result) * len(current_lines),
                    end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type,
                    symbol_name=f"{chunk.symbol_name}_part{len(result) + 1}"
                    if chunk.symbol_name
                    else None,
                    context=chunk.context,
                )
            )

        return result


def compute_file_hash(content: str) -> str:
    """Compute BLAKE3 hash of file content."""
    return blake3.blake3(content.encode("utf-8")).hexdigest()


def compute_chunk_hash(text: str, filepath: str, start_line: int) -> str:
    """Compute hash for a chunk (for deduplication)."""
    data = f"{filepath}:{start_line}:{text}"
    return blake3.blake3(data.encode("utf-8")).hexdigest()[:32]
