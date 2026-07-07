"""GUI component modules — reusable UI building blocks.

Public API — import rendering functions directly::

    from unjess.gui.components import render_message, render_approval_dialog
"""

from unjess.gui.components.agents_panel import render_agents_panel
from unjess.gui.components.approval_dialog import render_approval_dialog
from unjess.gui.components.chat_message import render_message
from unjess.gui.components.command_palette import render_command_palette
from unjess.gui.components.diff_view import render_diff
from unjess.gui.components.file_tree import render_file_tree
from unjess.gui.components.new_conversation_dialog import create_new_conversation_dialog
from unjess.gui.components.project_browser import render_project_browser
from unjess.gui.components.session_panel import render_session_panel
from unjess.gui.components.tool_card import render_tool_card

__all__ = [
    "create_new_conversation_dialog",
    "render_agents_panel",
    "render_approval_dialog",
    "render_command_palette",
    "render_diff",
    "render_file_tree",
    "render_message",
    "render_project_browser",
    "render_session_panel",
    "render_tool_card",
]
