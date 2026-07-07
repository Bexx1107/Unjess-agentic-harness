"""Workspace file-tree browser.

Renders a lazily-loaded directory tree using ``ui.tree``.  Only the
top-level directory is scanned initially; subdirectories are loaded on
expansion.  Common noise directories (``.git``, ``node_modules``, etc.)
are filtered out.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nicegui import ui


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Directories to hide from the tree.
IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "bower_components",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    ".tox", ".nox",
    "dist", "build", ".eggs", "*.egg-info",
    ".next", ".nuxt", ".output",
    ".idea", ".vscode",
    "target",  # Rust/Java build
})

#: Maximum children per directory before we stop showing individual files.
_MAX_CHILDREN = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_ignore(name: str) -> bool:
    """Check whether a file/directory name matches ignore patterns.

    Args:
        name: Bare file or directory name.

    Returns:
        ``True`` if the entry should be hidden.
    """
    return name in IGNORE_DIRS or name.startswith(".") or name.endswith(".egg-info")


def _scan_dir(dir_path: Path, lazy: bool = True) -> list[dict[str, Any]]:
    """Scan a directory and return tree nodes.

    Args:
        dir_path: Directory to scan.
        lazy: If ``True``, directories get an ``"_unloaded"`` child so
            ``ui.tree`` shows an expand arrow without scanning children.

    Returns:
        List of node dicts for ``ui.tree``.
    """
    nodes: list[dict[str, Any]] = []

    try:
        entries = sorted(
            dir_path.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except PermissionError:
        return nodes

    count = 0
    for entry in entries:
        if _should_ignore(entry.name):
            continue
        count += 1
        if count > _MAX_CHILDREN:
            nodes.append({
                "id": str(entry.parent / "…"),
                "label": f"… ({len(entries) - _MAX_CHILDREN} more entries)",
                "icon": "more_horiz",
            })
            break

        if entry.is_dir():
            node: dict[str, Any] = {
                "id": str(entry),
                "label": entry.name,
                "icon": "folder",
            }
            if lazy:
                # Placeholder child so the arrow shows
                node["children"] = [{"id": f"{entry}/_stub", "label": ""}]
            nodes.append(node)
        else:
            # Determine icon by extension
            icon = _file_icon(entry.suffix)
            nodes.append({
                "id": str(entry),
                "label": entry.name,
                "icon": icon,
            })

    return nodes


def _file_icon(suffix: str) -> str:
    """Map a file extension to a Material icon name.

    Args:
        suffix: File extension including the leading dot.

    Returns:
        Material icon name string.
    """
    mapping: dict[str, str] = {
        ".py": "code",
        ".js": "javascript",
        ".ts": "javascript",
        ".json": "data_object",
        ".yaml": "settings",
        ".yml": "settings",
        ".toml": "settings",
        ".md": "article",
        ".txt": "description",
        ".html": "html",
        ".css": "css",
        ".sh": "terminal",
        ".bat": "terminal",
        ".ps1": "terminal",
        ".rs": "code",
        ".go": "code",
        ".java": "code",
        ".c": "code",
        ".cpp": "code",
        ".h": "code",
    }
    return mapping.get(suffix.lower(), "description")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_file_tree(workspace: str) -> None:
    """Render an interactive workspace file browser.

    Creates a ``ui.tree`` rooted at *workspace*.  Subdirectories are
    loaded lazily when the user expands them.  Clicking a file shows
    its full path via ``ui.notify``.

    Args:
        workspace: Root directory path string.
    """
    root = Path(workspace).resolve()

    if not root.is_dir():
        ui.label(f"Workspace not found: {workspace}").classes(
            "text-sm text-red-5"
        )
        return

    # Initial scan — top level only
    initial_nodes = _scan_dir(root, lazy=True)
    if not initial_nodes:
        ui.label("Empty workspace").classes("text-sm text-grey-6")
        return

    root_node: list[dict[str, Any]] = [{
        "id": str(root),
        "label": root.name,
        "icon": "folder_open",
        "children": initial_nodes,
    }]

    tree = ui.tree(
        root_node,
        label_key="label",
        children_key="children",
        node_key="id",
    ).classes(
        "w-full file-tree"
    ).props('dense no-connectors selected-color="purple-4"')

    # --- Lazy loading on expand ---

    def _on_expand(e: Any) -> None:
        """Load children when a directory node is expanded."""
        node_id = e.args if isinstance(e.args, str) else (e.args or "")
        if not node_id:
            return

        node_path = Path(node_id)
        if not node_path.is_dir():
            return

        # Replace stub children with real children
        children = _scan_dir(node_path, lazy=True)
        _update_node_children(tree, node_id, children)

    def _on_select(e: Any) -> None:
        """Show file path when a file node is selected."""
        node_id = e.args if isinstance(e.args, str) else (e.args or "")
        if not node_id:
            return
        node_path = Path(node_id)
        if node_path.is_file():
            ui.notify(
                f"📄 {node_path}",
                type="info",
                position="bottom",
                timeout=3000,
            )

    tree.on("lazy-load", _on_expand)
    tree.on("update:selected", _on_select)


def _update_node_children(
    tree: Any,
    node_id: str,
    children: list[dict[str, Any]],
) -> None:
    """Replace stub children of a tree node with real children.

    This mutates the tree's internal node list.  NiceGUI's ``ui.tree``
    doesn't have a built-in lazy-load API, so we walk the props directly.

    Args:
        tree: The ``ui.tree`` element.
        node_id: ID of the parent node to update.
        children: New children list.
    """
    # Walk the tree data structure and update in place
    def _walk(nodes: list[dict]) -> bool:
        for node in nodes:
            if node.get("id") == node_id:
                node["children"] = children
                return True
            if "children" in node:
                if _walk(node["children"]):
                    return True
        return False

    if hasattr(tree, "_props") and "nodes" in tree._props:
        _walk(tree._props["nodes"])
        tree.update()
