"""Language configurations for tree-sitter parsing.

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

from pathlib import Path

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
from tree_sitter import Language

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
