"""System prompt builder — assembles the dynamic system prompt before each LLM call.

Now includes optional repo map, project context, user rules, and learned memory.
"""

import logging
from pathlib import Path
from typing import Optional

from unjess.config import Settings, get_os_info, get_shell
from unjess.tools import ToolRegistry

logger = logging.getLogger(__name__)


def build_system_prompt(
    settings: Settings,
    tool_registry: ToolRegistry,
    repo_map: str = "",
    project_context: str = "",
    user_rules: str = "",
    memory_context: str = "",
    planning_mode: bool = False,
    workspace: Path | None = None,
    gui_mode: bool = False,
    skill_context: str = "",
    user_profile_context: str = "",
) -> str:
    """Assemble the full system prompt from dynamic sections.

    Blocks:
        1. Identity
        2. User information
        3. Available tools
        4. Repo map (if indexed)
        5. Project context (if project_context.md exists)
        6. User rules (AGENTS.md + learned rules)
        7. Planning mode (if active)
        8. Memory (optional)
        9. Guidelines
        10. Communication style

    Args:
        settings: Current runtime settings.
        tool_registry: The tool registry (for tool descriptions).
        repo_map: Pre-formatted repo map string (from RepoMap.format()).
        project_context: Project context document content.
        user_rules: User rules from AGENTS.md and /learn.
        memory_context: Recent conversation summaries.
        planning_mode: Whether to inject plan-first instructions.
        gui_mode: Whether running in GUI/desktop app mode.
        skill_context: Skill catalog + matched skill instructions.

    Returns:
        The complete system prompt as a single string.
    """
    sections: list[str] = []

    # 1. Identity
    sections.append(_build_identity(gui_mode=gui_mode))

    # 2. User information
    sections.append(_build_user_info(settings, user_profile_context=user_profile_context))

    # 3. Available tools
    sections.append(_build_tools_section(tool_registry))

    # 3b. MCP server instructions (if any connected)
    mcp_instructions = _load_mcp_instructions(tool_registry)
    if mcp_instructions:
        sections.append(f"<mcp_instructions>\n{mcp_instructions}\n</mcp_instructions>")

    # 4. Repo map (optional — from codebase indexer)
    if repo_map:
        sections.append(f"<repo_map>\n{repo_map}\n</repo_map>")

    # 5. Project context (optional — from /init)
    if project_context:
        sections.append(f"<project_context>\n{project_context}\n</project_context>")

    # 6. User rules (optional — AGENTS.md + learned rules)
    if user_rules:
        sections.append(f"<user_rules>\n{user_rules}\n</user_rules>")


    # 6b. Skills (catalog + matched instructions)
    if skill_context:
        sections.append(f"<skills>\n{skill_context}\n</skills>")

    # 7. Planning mode (if active — for complex tasks)
    if planning_mode:
        sections.append(_build_planning_mode())

    # 8. Memory (optional — recent conversation summaries)
    if memory_context:
        sections.append(f"<memory>\n{memory_context}\n</memory>")

    # 9. Guidelines
    sections.append(_build_guidelines())

    # 10. Communication style
    sections.append(_build_communication_style())

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_identity(gui_mode: bool = False) -> str:
    """Block 1: Who am I?"""
    if gui_mode:
        environment = "the user's desktop app"
    else:
        environment = "the user's terminal"
    return f"""<identity>
You are Unjess (njss), a powerful AI coding agent running in {environment}.
You are pair programming with the user to solve their coding tasks.

Core capabilities:
- Read, write, and edit files in the user's project
- Run shell commands (with user approval via the permissions system)
- Search across the codebase (grep, file listing, repo map)
- Search the web (DuckDuckGo) and read URLs
- Browse web pages interactively (navigate, click, type, screenshot, JS eval)
- Ask the user clarifying questions with numbered options
- Manage background tasks

Advanced capabilities:
- Planning mode: For complex tasks, you plan first then execute in phases
- Goal mode: Extended autonomy for large tasks (up to 200 iterations)
- Subagent spawning: Delegate research or subtasks to background agents
- Cross-session memory: You remember learned rules and past conversation context
- Codebase awareness: You have access to a repo map showing file structure and symbols

Slash commands the user can use:
- /help — show commands  |  /model — switch model  |  /clear — clear history
- /goal <task> — run with extended autonomy  |  /plan — toggle planning mode
- /browser [url] — browse the web  |  /test [cmd] — run tests
- /diff — show session changes  |  /undo — rollback changes
- /compact — compress conversation  |  /context — show token usage
- /cost — session costs  |  /budget <$> — set cost limit
- /settings — view/change config  |  /setup — reconfigure provider
- /learn <rule> — teach a persistent rule  |  /free — free-tier mode
- /map — show codebase structure  |  /init — generate project context
- /agents — manage subagents  |  /schedule — timers

IMPORTANT — When to use tools vs. just talk:
- For greetings (hi, hello, hey, etc.) — just respond with friendly text. Do NOT call any tools.
- For simple questions — just answer in text. Do NOT call any tools.
- For coding tasks — use the appropriate tools (read_file, write_file, run_command, etc.).
- NEVER use ask_question just to greet the user or make casual conversation.

You always prioritize the user's request. If a task is unclear, use the ask_question tool.
</identity>"""


def _build_user_info(settings: Settings, user_profile_context: str = "") -> str:
    """Block 2: Dynamic user/environment info."""
    workspace = Path(settings.workspace).resolve()
    os_name = get_os_info()
    shell = get_shell()
    model = settings.model
    provider = getattr(settings, 'provider', 'unknown')

    profile_block = f"\nUser Persona / Cross-Session Profile:\n{user_profile_context}\n" if user_profile_context else ""

    os_specific_block = ""
    if "windows" in os_name.lower():
        os_specific_block = (
            "\nWINDOWS POWERSHELL EXECUTION RULES:\n"
            "- NEVER use bash syntax or variables like $HOME or ~ in commands or cwd. Use Windows paths.\n"
            "- NEVER use 'cd' inside run_command. Pass the target directory using the 'cwd' parameter.\n"
            "- Use native PowerShell cmdlets or cross-platform commands.\n"
        )

    return f"""<user_information>
Operating System: {os_name}
Shell: {shell}
Workspace: {workspace}
Current Model: {model}
Provider: {provider}
{profile_block}
You are running as model "{model}" via the "{provider}" provider.
All file paths should be relative to the workspace root: {workspace}
When running commands, use the appropriate shell syntax for {shell} on {os_name}.
{os_specific_block}</user_information>"""


def _build_tools_section(registry: ToolRegistry) -> str:
    """Block 3: Auto-generated tool descriptions."""
    tools = registry.get_tools()

    if not tools:
        return "<tools>\nNo tools available.\n</tools>"

    lines = ["<tools>", "You have access to the following tools:", ""]

    for tool in tools:
        name = tool["name"]
        desc = tool.get("description", "")
        params = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", [])

        # Build parameter list
        param_parts: list[str] = []
        for pname, pschema in params.items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            req = " (required)" if pname in required else ""
            param_parts.append(f"    - {pname}: {ptype}{req} — {pdesc}")

        lines.append(f"**{name}**: {desc}")
        if param_parts:
            lines.append("  Parameters:")
            lines.extend(param_parts)
        lines.append("")

    lines.append("To use a tool, output a function call. You may call multiple tools in sequence.")
    lines.append("All file paths are relative to the workspace root.")
    lines.append("</tools>")

    return "\n".join(lines)


def _build_guidelines() -> str:
    """Block 9: Behavioral guidelines."""
    return """<guidelines>
## File editing
- Always read files before editing them to understand the current state.
- Limit reconnaissance: Read ONLY the files you strictly need (1 to 2 turns max). Do NOT get stuck reading files indefinitely before answering or acting.
- When using `edit_file`, use an exact unique code snippet for `target` from the file.
- When editing files, preserve existing comments and code style.
- After making changes, briefly explain what you did and why.
- If a command might be destructive, warn the user.

## Planning — create plan.md for complex tasks
When the user asks you to BUILD something non-trivial (a new app, feature, refactor, migration):
1. **Create `plan.md`** in the workspace root with your implementation plan.
   - Break the work into numbered phases.
   - List which files will be created/modified/deleted per phase.
   - Include key design decisions and trade-offs.
2. **Tell the user** the plan is ready and ask for approval before coding.
3. **Execute phase by phase** after approval, updating plan.md as you go.
4. Do NOT plan for simple fixes, one-file edits, or quick questions.

## Reports — write .md files for research and audits
When the user asks you to RESEARCH, AUDIT, ANALYZE, or REVIEW something:
1. **Create a report file** (e.g. `audit.md`, `research.md`, `analysis.md`) in the workspace.
2. Write a well-structured markdown document with headings, tables, and findings.
3. Tell the user the file is ready and highlight key findings in chat.
4. Do NOT dump a wall of text into the chat — put it in the file.

## Tool usage rules
- NEVER use `cd` inside `run_command`. Specify the target directory using the `cwd` parameter instead.
- NEVER use `run_command` with inline `python -c "..."` to read files, search text, or list directory contents. ALWAYS use dedicated native tools (`view_file`, `list_dir`, `grep_search`).
- If a tool or command returns empty output or fails, NEVER panic, make up data, argue, or fabricate harness errors. Methodically inspect the state with `list_dir` or `view_file` and retry cleanly.
- Do NOT automatically use browser tools unless the user explicitly asks to test or browse.
- Do NOT automatically run servers, apps, or test suites. Tell the user the command and ask.
- Do NOT take screenshots or navigate to localhost unless asked.
- When you finish building something, summarize what you built and suggest how to test it.
- Use the ask_question tool ONLY when you need the user to make a multi-choice decision.
- NEVER use ask_question for greetings, casual conversation, or open-ended questions.
- If you want to say "How can I help?" — just output that as text, do NOT call ask_question.

## General
- Keep responses focused and concise.
- If you're unsure, say so rather than guessing.
</guidelines>"""


def _build_communication_style() -> str:
    """Block 10: Response formatting."""
    return """<communication_style>
- Format responses in markdown when helpful.
- Use code blocks with language identifiers for code snippets.
- Keep explanations brief unless the user asks for detail.
- When showing file changes, include the filename.
</communication_style>"""


def _build_planning_mode() -> str:
    """Block for complex tasks — instructs the LLM to plan before executing."""
    return """<planning_mode>
This task has been identified as complex. You MUST follow the plan-first workflow:

1. **Understand**: Inspect ONLY 1 to 3 essential files directly related to the user's prompt (at most 1-2 turns of reading).
   - Do NOT conduct an exhaustive repository exploration or read every file.
   - Gather minimal required context and move immediately to planning.

2. **Plan**: Present your phased implementation plan directly in your text response to the user.
   - Break the work into numbered phases (Phase 1, Phase 2, etc.).
   - Each phase should be small enough to review independently.
   - List which files will be created, modified, or deleted in each phase.
   - Explain the approach and key design decisions.
   - Stop calling tools and output your plan to the user!

3. **Confirm**: Ask the user to approve the plan before proceeding.
   - Say: "Does this plan look good? I'll start with Phase 1."
   - Wait for explicit approval (e.g. "go", "yes", "looks good").

4. **Execute**: Implement one phase at a time.
   - After completing each phase, summarize what was done.
   - Ask before moving to the next phase if it's a large change.

IMPORTANT:
- Do NOT get trapped in an endless read loop. 1-2 turns of reading is plenty before presenting your plan.
- NEVER start writing code for complex tasks without presenting a plan first.
- NEVER implement everything in one massive batch — phase it out.
- Simple follow-ups and small fixes do NOT need a plan.
</planning_mode>"""


def load_project_context(workspace: Path) -> str:
    """Load project_context.md if it exists.

    Args:
        workspace: Project root.

    Returns:
        Content string, or empty string.
    """
    ctx_path = workspace / "project_context.md"
    if ctx_path.exists():
        try:
            return ctx_path.read_text(encoding="utf-8", errors="replace")[:4000]
        except Exception:
            logger.debug("Failed to load project context from %s", ctx_path, exc_info=True)
    return ""


def load_user_rules(workspace: Path) -> str:
    """Load AGENTS.md rules from workspace and global config.

    Args:
        workspace: Project root.

    Returns:
        Combined rules string.
    """
    rules_parts: list[str] = []

    # Project-level rules
    project_rules = workspace / ".agents" / "AGENTS.md"
    if project_rules.exists():
        try:
            rules_parts.append(project_rules.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            logger.debug("Failed to load project rules from %s", project_rules, exc_info=True)

    # Global rules
    global_rules = Path.home() / ".unjess" / "AGENTS.md"
    if global_rules.exists():
        try:
            rules_parts.append(global_rules.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            logger.debug("Failed to load global rules from %s", global_rules, exc_info=True)

    return "\n\n".join(rules_parts)


def _load_agents_rules(workspace: Path) -> str:
    """Load user rules from AGENTS.md files.

    Checks:
    1. Workspace: .agents/AGENTS.md
    2. Global: ~/.unjess/AGENTS.md
    """
    rules: list[str] = []

    # Workspace rules
    workspace_rules = workspace / ".agents" / "AGENTS.md"
    if workspace_rules.is_file():
        try:
            content = workspace_rules.read_text(encoding="utf-8").strip()
            if content:
                rules.append(f"### Workspace Rules\n{content}")
        except Exception:
            pass

    # Global rules
    global_rules = Path.home() / ".unjess" / "AGENTS.md"
    if global_rules.is_file():
        try:
            content = global_rules.read_text(encoding="utf-8").strip()
            if content:
                rules.append(f"### Global Rules\n{content}")
        except Exception:
            pass

    if not rules:
        return ""
    return "## User Rules\n\n" + "\n\n".join(rules)


def _load_mcp_instructions(tool_registry: ToolRegistry) -> str:
    """Load MCP server instructions for any connected servers.

    Checks for instruction files in ``~/.unjess/mcp_instructions/<name>.md``
    by scanning tool names for ``mcp_<servername>_`` prefixes.
    """
    instructions_dir = Path.home() / ".unjess" / "mcp_instructions"
    if not instructions_dir.exists():
        return ""

    # Extract server names from registered MCP tool names
    server_names: set[str] = set()
    for name in tool_registry.tool_names:
        if name.startswith("mcp_"):
            # Format: mcp_<servername>_<toolname>
            parts = name.split("_", 2)
            if len(parts) >= 3:
                server_names.add(parts[1])

    if not server_names:
        return ""

    blocks: list[str] = []
    for sname in sorted(server_names):
        # Try exact match and hyphenated variants
        for candidate in [f"{sname}.md", f"{sname.replace('_', '-')}.md"]:
            instr_file = instructions_dir / candidate
            if instr_file.exists():
                try:
                    content = instr_file.read_text(encoding="utf-8").strip()
                    if content:
                        blocks.append(content)
                except Exception:
                    pass
                break  # found one, don't try variants

    return "\n\n".join(blocks)
