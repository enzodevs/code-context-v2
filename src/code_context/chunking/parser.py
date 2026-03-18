"""AST-based hierarchical code parser using tree-sitter.

Implements 3-level hierarchical chunking:
1. file-level: Complete file for broad context
2. declaration-level: Top-level exports/declarations (const, class, interface)
3. function-level: Individual functions/methods for precise retrieval
"""

import logging
from pathlib import Path

import blake3
from tree_sitter import Node, Parser

from code_context.chunking.languages import LANGUAGE_CONFIG, detect_language
from code_context.chunking.models import (
    ParsedChunk,
    _encoder,
    compute_chunk_hash,
    compute_file_hash,
)
from code_context.config import get_settings

logger = logging.getLogger(__name__)


class CodeParser:
    """Multi-language hierarchical code parser using tree-sitter.

    Extracts chunks at multiple levels depending on language:
    - Code files: file -> declaration -> function/method
    - SQL: file -> statement
    - Markdown: file -> section
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
        source_bytes = content.encode("utf-8")
        tree = parser.parse(source_bytes)

        config = LANGUAGE_CONFIG[language]
        strategy = config.get("chunk_strategy", "code")

        chunks: list[ParsedChunk] = []

        # Collect imports for context (if applicable)
        imports = []
        if "import" in config.get("node_types", {}):
            imports = self._collect_imports(
                tree.root_node, source_bytes, config["node_types"].get("import", set())
            )

        # Track whether this is a small file (used for file-chunk preference rules)
        line_count = content.count("\n") + 1
        is_small_file = line_count <= self.settings.small_file_lines

        # Level 1: File-level chunk (complete file)
        file_chunk = self._create_file_chunk(filepath, content, imports, tree.root_node)
        if file_chunk:
            chunks.append(file_chunk)

        # Strategy-specific chunking
        if strategy == "code":
            # Level 2: Declaration-level chunks
            declaration_chunks = self._extract_declarations(
                tree.root_node, source_bytes, language, filepath, imports
            )
            chunks.extend(declaration_chunks)

            # Level 3: Function/method-level chunks
            function_chunks = self._extract_functions(
                tree.root_node, source_bytes, language, filepath, imports
            )
            chunks.extend(function_chunks)

        elif strategy == "statement":
            # SQL: Extract statements
            statement_chunks = self._extract_statements(
                tree.root_node, source_bytes, language, filepath
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
                if self.settings.ast_split_enabled:
                    sub_chunks = self._split_chunk_structurally(chunk, max_tokens, language)
                else:
                    sub_chunks = self._split_large_chunk(chunk, max_tokens)
                filtered.extend(c for c in sub_chunks if c.token_count >= min_tokens)
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

        # For small code files, prefer symbol-scoped chunks over generic file chunk
        # whenever we extracted any declarations/functions/methods.
        if strategy == "code" and is_small_file:
            has_symbol_chunks = any(
                c.chunk_type in {"declaration", "function", "method", "class"}
                for c in deduplicated
            )
            if has_symbol_chunks:
                deduplicated = [c for c in deduplicated if c.chunk_type != "file"]

        return deduplicated

    def _create_file_chunk(
        self, filepath: str, content: str, imports: list[str], root_node: Node | None = None,
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
            _token_count=token_count,
            _node=root_node,
        )

    def _extract_declarations(
        self,
        root: Node,
        source: bytes,
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

            name = self._get_declaration_name(child, config.get("name_field"))
            text = source[child.start_byte:child.end_byte].decode("utf-8")

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

    def _get_declaration_name(self, node: Node, name_field: str | None) -> str | None:
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
        source: bytes,
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
        source: bytes,
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
                        text = source[child.start_byte:child.end_byte].decode("utf-8")
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

    def _extract_sql_name(self, node: Node, source: bytes) -> str | None:
        """Extract name from SQL statement (table name, etc.)."""
        # Look for identifier after CREATE TABLE, CREATE INDEX, etc.
        for child in node.children:
            if child.type in {"identifier", "table_reference", "object_reference"}:
                return source[child.start_byte:child.end_byte].decode("utf-8")
            # Recurse one level for nested identifiers
            for subchild in child.children:
                if subchild.type == "identifier":
                    return source[subchild.start_byte:subchild.end_byte].decode("utf-8")
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
        self, root: Node, source: bytes, import_types: set[str]
    ) -> list[str]:
        """Collect import statements from the file."""
        imports = []

        def visit(node: Node):
            if node.type in import_types:
                text = source[node.start_byte : node.end_byte].decode("utf-8").strip()
                imports.append(text)
            for child in node.children:
                visit(child)

        visit(root)
        return imports

    def _get_node_name(self, node: Node, name_field: str | None) -> str | None:
        """Extract the name from a node.

        For arrow functions / function expressions nested inside call_expression
        (e.g. Convex ``export const foo = query({ handler: async () => {...} })``),
        the name lives on an ancestor ``variable_declarator``, not on the function
        node itself.  We walk up to 6 parent levels to find it.
        """
        if not name_field:
            return None

        name_node = node.child_by_field_name(name_field)
        if name_node:
            return name_node.text.decode("utf-8") if name_node.text else None

        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8") if child.text else None

        # Ancestry walk: arrow_function / function_expression inside a
        # variable_declarator (possibly via call_expression, pair, object, etc.)
        if node.type in {"arrow_function", "function_expression"}:
            parent = node.parent
            for _ in range(6):
                if parent is None:
                    break
                if parent.type == "variable_declarator":
                    vname = parent.child_by_field_name("name")
                    if vname:
                        return vname.text.decode("utf-8") if vname.text else None
                    break
                parent = parent.parent

        return None

    def _node_to_chunk(
        self,
        node: Node,
        source: bytes,
        chunk_type: str,
        symbol_name: str | None,
        filepath: str,
        imports: list[str],
        parent_class: str | None = None,
    ) -> ParsedChunk | None:
        """Convert a tree-sitter node to a ParsedChunk."""
        text = source[node.start_byte : node.end_byte].decode("utf-8")
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

        # Extract signature for functions/methods (text before the body block)
        if chunk_type in ("function", "method"):
            body = node.child_by_field_name("body")
            if body:
                sig = source[node.start_byte : body.start_byte].decode("utf-8").strip()
                if sig:
                    context["signature"] = sig

        return ParsedChunk(
            text=text,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            symbol_name=symbol_name,
            context=context,
            _node=node,
        )

    # Node types to skip when collecting structural children for splitting
    _SKIP_NODE_TYPES = frozenset({
        "comment", "block_comment", "line_comment", "{", "}", "(", ")", ";",
        ",", ":", "=>", ".", "//", "/*", "*/",
    })

    def _split_chunk_structurally(
        self,
        chunk: ParsedChunk,
        max_tokens: int,
        language: str,
        _depth: int = 0,
    ) -> list[ParsedChunk]:
        """Split an oversized chunk using AST structure instead of blind token slicing.

        Strategy (3-level fallback):
        1. If AST node available, split at structural children (statements, methods).
        2. Recurse on oversized sub-groups (depth limit = 3).
        3. Fall back to line-boundary splitting as last resort.
        """
        node = chunk._node
        if node is None or _depth >= 3:
            return self._split_by_lines(chunk, max_tokens)

        # Get the node whose children we should split on.
        # For functions/methods, we want the body's children (statements).
        target = node
        body = node.child_by_field_name("body")
        if body is not None:
            target = body

        children = [
            c for c in target.children
            if c.type not in self._SKIP_NODE_TYPES and c.end_byte > c.start_byte
        ]

        if not children:
            return self._split_by_lines(chunk, max_tokens)

        # Extract the source bytes that corresponds to this chunk
        source = chunk.text.encode("utf-8")
        source_offset = chunk._node.start_byte if chunk._node else 0

        groups = self._group_children_by_budget(
            children, source_offset, source, chunk, max_tokens,
        )

        result: list[ParsedChunk] = []
        for group_chunk in groups:
            if group_chunk.token_count > max_tokens:
                result.extend(
                    self._split_chunk_structurally(group_chunk, max_tokens, language, _depth + 1)
                )
            else:
                result.append(group_chunk)
        return result

    def _group_children_by_budget(
        self,
        children: list[Node],
        source_offset: int,
        source: bytes,
        parent_chunk: ParsedChunk,
        max_tokens: int,
    ) -> list[ParsedChunk]:
        """Group adjacent AST children into chunks that fit within token budget."""
        groups: list[ParsedChunk] = []
        current_nodes: list[Node] = []
        current_tokens = 0

        for child in children:
            child_text = source[child.start_byte - source_offset : child.end_byte - source_offset].decode("utf-8")
            child_tokens = len(_encoder.encode(child_text))

            if current_nodes and current_tokens + child_tokens > max_tokens:
                group = self._nodes_to_chunk(
                    current_nodes, source_offset, source, parent_chunk, current_tokens,
                )
                if group:
                    groups.append(group)
                current_nodes = []
                current_tokens = 0

            current_nodes.append(child)
            current_tokens += child_tokens

        if current_nodes:
            group = self._nodes_to_chunk(
                current_nodes, source_offset, source, parent_chunk, current_tokens,
            )
            if group:
                groups.append(group)

        return groups

    def _nodes_to_chunk(
        self,
        nodes: list[Node],
        source_offset: int,
        source: bytes,
        parent_chunk: ParsedChunk,
        token_count: int,
    ) -> ParsedChunk | None:
        """Create a ParsedChunk from a group of adjacent AST nodes."""
        if not nodes:
            return None

        start_byte = nodes[0].start_byte - source_offset
        end_byte = nodes[-1].end_byte - source_offset
        text = source[start_byte:end_byte].decode("utf-8")

        if not text.strip():
            return None

        start_line = nodes[0].start_point[0] + 1
        end_line = nodes[-1].end_point[0] + 1

        symbol_name = parent_chunk.symbol_name

        return ParsedChunk(
            text=text,
            start_line=start_line,
            end_line=end_line,
            chunk_type=parent_chunk.chunk_type,
            symbol_name=symbol_name,
            context=parent_chunk.context,
            _token_count=token_count,
            _node=nodes[0] if len(nodes) == 1 else None,
        )

    def _split_by_lines(
        self, chunk: ParsedChunk, max_tokens: int
    ) -> list[ParsedChunk]:
        """Split a chunk at line boundaries (better than blind token split)."""
        lines = chunk.text.split("\n")
        result: list[ParsedChunk] = []
        current_lines: list[str] = []
        current_tokens = 0
        lines_emitted = 0

        def _emit():
            nonlocal current_lines, current_tokens, lines_emitted
            text = "\n".join(current_lines)
            part_num = len(result) + 1
            start_line = chunk.start_line + lines_emitted

            result.append(ParsedChunk(
                text=text,
                start_line=start_line,
                end_line=start_line + len(current_lines) - 1,
                chunk_type=chunk.chunk_type,
                symbol_name=f"{chunk.symbol_name}_part{part_num}" if chunk.symbol_name else None,
                context=chunk.context,
                _token_count=current_tokens,
            ))
            lines_emitted += len(current_lines)
            current_lines = []
            current_tokens = 0

        for line in lines:
            line_tokens = len(_encoder.encode(line))

            if current_lines and current_tokens + line_tokens > max_tokens:
                _emit()

            current_lines.append(line)
            current_tokens += line_tokens

        if current_lines:
            _emit()

        return result if result else [chunk]

    def _split_large_chunk(
        self, chunk: ParsedChunk, max_tokens: int
    ) -> list[ParsedChunk]:
        """Split a large chunk into smaller pieces.

        Encodes the full text once, splits the token array into slices
        of max_tokens, and decodes each slice back to text.
        """
        all_tokens = _encoder.encode(chunk.text)

        if len(all_tokens) <= max_tokens:
            return [chunk]

        result = []
        lines_consumed = 0

        for i in range(0, len(all_tokens), max_tokens):
            token_slice = all_tokens[i : i + max_tokens]
            text = _encoder.decode(token_slice)

            part_num = len(result) + 1
            start_line = chunk.start_line + lines_consumed
            new_lines = text.count("\n")
            end_line = start_line + new_lines
            lines_consumed += new_lines

            result.append(
                ParsedChunk(
                    text=text,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=chunk.chunk_type,
                    symbol_name=f"{chunk.symbol_name}_part{part_num}"
                    if chunk.symbol_name
                    else None,
                    context=chunk.context,
                    _token_count=len(token_slice),
                )
            )

        return result
