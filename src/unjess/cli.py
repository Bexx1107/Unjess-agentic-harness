"""CLI entry point and REPL loop.

Handles argument parsing, initialization, and the interactive session.
Wires up ALL components from Phases 1-6.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

# Force UTF-8 encoding on Windows (PyInstaller windowed apps default to charmap/cp1252)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

# Patch asyncio to allow nested event loops (needed when Playwright + prompt_toolkit coexist)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from unjess import __version__
from unjess.agent import Agent
from unjess.commands import CommandContext, dispatch
from unjess.config import (
    Settings,
    apply_cli_overrides,
    ensure_config,
    load_config,
    resolve_workspace,
)
from unjess.context_manager import ContextManager
from unjess.conversation_logger import ConversationLogger
from unjess.display import Display, AGENT_THEME
from unjess.git_integration import GitIntegration
from unjess.indexer import RepoMap
from unjess.llm.router import ProviderRouter
from unjess.memory import MemoryStore
from unjess.mentions import MentionResolver
from unjess.permissions import PermissionManager
from unjess.tools import ToolRegistry
from unjess.tools.file_tools import register_file_tools
from unjess.tools.search_tools import register_search_tools
from unjess.tools.command_tools import register_command_tools
from unjess.tools.web_tools import register_web_tools
from unjess.tools.browser_tools import register_browser_tools
from unjess.sandbox import Sandbox
from unjess.undo import UndoManager
from unjess.workspace import detect_project_root, detect_project_type
from unjess.auto_summary import generate_summary
from unjess.embeddings import get_embedding_provider
from unjess.knowledge_graph import KnowledgeGraph
from unjess.rag import RAGEngine
from unjess.skills.engine import SkillEngine
from unjess.vector_store import VectorStore

if TYPE_CHECKING:
    from unjess.protocols import InputProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TerminalInput — InputProtocol implementation for CLI mode
# ---------------------------------------------------------------------------

class TerminalInput:
    """Terminal-based input handler implementing InputProtocol.

    Wraps ``prompt_toolkit`` for the REPL and ``rich`` for approval prompts.
    """

    def __init__(self, console: Console) -> None:
        self._console = console

    def get_user_input(self, prompt: str = "❯ ") -> str:
        """Get input from the terminal."""
        return input(prompt)

    def ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, bool]:
        """Show an approval prompt with rich panel."""
        if tool_name == "run_command":
            command = args.get("command", "<unknown>")
            cwd = args.get("cwd", ".")
            preview_text = Text()
            preview_text.append("Command: ", style="bold")
            preview_text.append(command, style="bold yellow")
            preview_text.append(f"\n     cwd: {cwd}", style="dim")
        else:
            preview_text = Text(f"{tool_name}({args})")

        panel = Panel(
            preview_text,
            title="⚠️  Permission Required",
            title_align="left",
            border_style="yellow",
        )
        self._console.print(panel)

        while True:
            try:
                self._console.print(
                    "[bold]Allow?[/bold] [dim](y)es / (n)o / (a)lways[/dim]: ",
                    end="",
                )
                response = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                self._console.print()
                return (False, False)

            if response in ("y", "yes"):
                return (True, False)
            if response in ("n", "no"):
                return (False, False)
            if response in ("a", "always"):
                return (True, True)

            self._console.print("  [dim]Please enter y, n, or a.[/dim]")

    def ask_question(self, question: str, options: list[str]) -> str:
        """Ask a question with numbered options."""
        self._console.print(f"\n  [bold yellow]❓ {question}[/bold yellow]")
        for i, opt in enumerate(options, 1):
            self._console.print(f"    {i}. {opt}")
        self._console.print()
        try:
            answer = input("  Your choice: ").strip()
            if answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            return answer
        except (EOFError, KeyboardInterrupt):
            return "(user cancelled)"

    def ask_confirmation(self, message: str) -> bool:
        """Ask a yes/no question."""
        self._console.print(f"  [bold]{message}[/bold] [dim](y/n)[/dim]: ", end="")
        try:
            response = input().strip().lower()
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def ask_choice(self, title: str, choices: list[tuple[str, str]]) -> str:
        """Show a numbered list of choices."""
        self._console.print(f"\n  [bold]{title}[/bold]")
        for i, (value, label) in enumerate(choices, 1):
            self._console.print(f"    {i}. {label}")
        self._console.print()
        try:
            answer = input("  Your choice (number): ").strip()
            if answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx < len(choices):
                    return choices[idx][0]
            return ""
        except (EOFError, KeyboardInterrupt):
            return ""


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _run_repl(
    agent: Agent,
    cmd_ctx: CommandContext,
    display: Display,
) -> None:
    """Run the interactive REPL loop."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.document import Document
        from prompt_toolkit.history import FileHistory

        from unjess.commands import get_all_commands

        # Known subcommands for commands that accept them
        _SUBCOMMANDS: dict[str, list[tuple[str, str]]] = {
            "/model": [
                ("all", "List ALL models from all providers"),
                ("list", "List ALL models from all providers"),
            ],
            "/free": [
                ("off", "Turn off free mode"),
                ("status", "Show free-tier usage stats"),
            ],
            "/provider": [
                ("google", "Switch to Google Gemini"),
                ("groq", "Switch to Groq"),
                ("mistral", "Switch to Mistral"),
                ("openrouter", "Switch to OpenRouter"),
                ("cerebras", "Switch to Cerebras"),
                ("xai", "Switch to xAI (Grok)"),
                ("openai", "Switch to OpenAI"),
                ("anthropic", "Switch to Anthropic"),
                ("ollama", "Switch to Ollama (local)"),
            ],
            "/undo": [
                ("list", "List recent operations"),
            ],
            "/settings": [
                ("model", "View/change model"),
                ("provider", "View/change provider"),
            ],
            "/plan": [
                ("on", "Enable plan-first mode"),
                ("off", "Disable plan-first mode"),
            ],
            "/mcp": [
                ("list", "List MCP servers"),
                ("connect", "Connect to an MCP server"),
                ("disconnect", "Disconnect an MCP server"),
                ("tools", "List MCP tools"),
            ],
            "/agents": [
                ("list", "List active subagents"),
                ("kill", "Kill a subagent"),
            ],
        }

        class SlashCommandCompleter(Completer):
            """Autocomplete for slash commands and their subcommands.

            Activates when the input starts with '/' and shows all
            matching commands with their descriptions. After a space,
            shows available subcommands for the current command.
            """

            def get_completions(
                self, document: Document, complete_event: object
            ) -> "Iterable[Completion]":
                text = document.text_before_cursor.lstrip()
                if not text.startswith("/"):
                    return

                # Check if we're completing a subcommand (has a space after the command)
                parts = text.split(None, 1)
                if len(parts) == 2:
                    cmd = parts[0]
                    sub_text = parts[1]
                    if cmd in _SUBCOMMANDS:
                        for sub_name, sub_desc in _SUBCOMMANDS[cmd]:
                            if sub_name.startswith(sub_text.lower()):
                                yield Completion(
                                    sub_name,
                                    start_position=-len(sub_text),
                                    display=f"{cmd} {sub_name}",
                                    display_meta=sub_desc,
                                )
                    return

                # Check if cursor is right after a known command + space
                if text.endswith(" ") and text.strip() in _SUBCOMMANDS:
                    cmd = text.strip()
                    for sub_name, sub_desc in _SUBCOMMANDS[cmd]:
                        yield Completion(
                            sub_name,
                            start_position=0,
                            display=f"{cmd} {sub_name}",
                            display_meta=sub_desc,
                        )
                    return

                # Top-level command completion
                commands = get_all_commands()
                for name, cmd in sorted(commands.items()):
                    if name.startswith(text):
                        usage = f"  {cmd.usage}" if cmd.usage else ""
                        yield Completion(
                            name,
                            start_position=-len(text),
                            display=f"{name}{usage}",
                            display_meta=cmd.description,
                        )

        history_path = Path.home() / ".unjess" / "history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        from prompt_toolkit.formatted_text import HTML

        def _bottom_toolbar() -> HTML:
            """Dynamic bottom toolbar showing model, cost, and status."""
            parts: list[str] = []

            # Model & provider
            model = cmd_ctx.settings.model or "?"
            provider = cmd_ctx.router.active_provider_name
            # Shorten long model names
            if len(model) > 25:
                model = model[:22] + "…"
            parts.append(f"<b>{model}</b> <style fg='ansibrightblack'>via {provider}</style>")

            # Free mode indicator
            if getattr(cmd_ctx.settings, "free_mode_enabled", False):
                parts.append("<style fg='ansigreen'>🆓 FREE</style>")

            # Cost from logger
            if cmd_ctx.logger:
                try:
                    ct = cmd_ctx.logger.cost_tracker
                    total_cost = ct.total_cost
                    total_tokens = ct.total_tokens_in + ct.total_tokens_out
                    tok_str = f"{total_tokens // 1000}K" if total_tokens >= 1000 else str(total_tokens)
                    if total_cost > 0:
                        parts.append(f"<style fg='ansiyellow'>${total_cost:.4f}</style> ({tok_str} tok)")
                    else:
                        parts.append(f"<style fg='ansibrightblack'>{tok_str} tok</style>")
                except Exception:
                    pass

            # Active agents count
            if hasattr(cmd_ctx.agent, '_subagent_manager'):
                try:
                    active = cmd_ctx.agent._subagent_manager.active_count
                    if active > 0:
                        parts.append(f"<style fg='ansicyan'>⚡ {active} agent{'s' if active > 1 else ''}</style>")
                except Exception:
                    pass

            return HTML(" │ ".join(parts))

        session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            completer=SlashCommandCompleter(),
            complete_while_typing=True,   # auto-show completions as you type /
            complete_in_thread=True,       # don't block input while completing
            multiline=False,
            bottom_toolbar=_bottom_toolbar,
        )
        use_prompt_toolkit = True
    except ImportError:
        use_prompt_toolkit = False
        session = None  # type: ignore[assignment]

    while True:
        try:
            if use_prompt_toolkit and session is not None:
                try:
                    user_input = session.prompt("❯ ")
                except RuntimeError as exc:
                    if "cannot be called from a running event loop" in str(exc):
                        # Playwright left an event loop running — fall back to basic input
                        logger.debug("prompt_toolkit asyncio conflict, using input(): %s", exc)
                        user_input = input("❯ ")
                    else:
                        raise
            else:
                user_input = input("❯ ")
        except (EOFError, KeyboardInterrupt):
            display.console.print()
            display.show_info("Goodbye! 👋")
            break

        user_input = user_input.strip()

        if not user_input:
            continue

        # Exit commands
        if user_input.lower() in ("exit", "quit"):
            display.show_info("Goodbye! 👋")
            break

        # Slash commands → dispatch to commands.py registry
        if user_input.startswith("/"):
            should_continue = dispatch(user_input, cmd_ctx)
            if not should_continue:
                break
            continue

        # Normal message → agent loop
        try:
            agent.run(user_input)
        except KeyboardInterrupt:
            display.console.print()
            display.show_info("Interrupted. Type /exit to quit.")
        except Exception as exc:
            display.show_error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main CLI entry point — parses args, initializes components, runs the agent."""
    parser = argparse.ArgumentParser(
        prog="njss",
        description="Unjess — AI coding agent",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="One-shot prompt (if provided, runs once and exits)",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Model to use (e.g. gpt-4o, claude-sonnet-4-20250514, gemini-2.5-flash)",
    )
    parser.add_argument(
        "-p", "--provider",
        default=None,
        help="Force a specific provider (openai, anthropic, google, ollama)",
    )
    parser.add_argument(
        "-w", "--workspace",
        default=None,
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: ~/.unjess/config.yaml)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"njss {__version__}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-approve all commands (dangerous!)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch NiceGUI web interface instead of terminal REPL",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Open GUI in a native desktop window (requires pywebview)",
    )

    args = parser.parse_args()


    # --- Logging ---
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(name)s: %(message)s')

    # --- Config ---
    config_path = Path(args.config) if args.config else None
    ensure_config(config_path)
    settings = load_config(config_path)

    # --- First-run wizard (if no model configured and no CLI override) ---
    # In GUI mode, skip — the GUI has its own onboarding page at /onboarding
    if not settings.model and args.model is None and not getattr(args, 'gui', False):
        from unjess.first_run import run_first_setup
        setup_console = Console()
        settings = run_first_setup(settings, setup_console)

    apply_cli_overrides(
        settings,
        model=args.model,
        provider=args.provider,
        workspace=args.workspace,
    )

    # --- Workspace detection ---
    if args.workspace:
        workspace = Path(args.workspace).resolve()
    elif getattr(args, 'gui', False):
        # GUI mode: default to home dir — workspace picker handles project selection
        workspace = Path.home()
    else:
        # CLI mode: use CWD and walk up to find project root
        workspace = Path.cwd()
        detected_root = detect_project_root(workspace)
        workspace = detected_root
    settings.workspace = str(workspace)

    # Detect project type
    project_info = detect_project_type(workspace)

    # --- Console + Display ---
    console = Console(theme=AGENT_THEME)
    display = Display(
        console=console,
        verbose=args.verbose or settings.verbosity == "verbose",
    )

    # --- Input Handler ---
    terminal_input: TerminalInput | None = None
    if not getattr(args, 'gui', False):
        terminal_input = TerminalInput(console)

    # --- Permissions ---
    permissions = PermissionManager(console=console, input_handler=terminal_input)
    if args.yes:
        permissions.approve_all_commands()

    # --- LLM Router ---
    try:
        router = ProviderRouter(settings)
    except Exception as exc:
        if getattr(args, 'gui', False):
            # GUI mode — don't crash; user will configure via onboarding
            logger.warning("ProviderRouter init failed (will configure via GUI): %s", exc)
            router = ProviderRouter.__new__(ProviderRouter)
            router._providers = {}
            router._settings = settings
            router._current_provider = None
            router._fallback_chain = []
        else:
            display.show_error(f"Failed to initialize LLM providers: {exc}")
            sys.exit(1)

    # --- Tools ---
    sandbox = Sandbox()
    tool_registry = ToolRegistry()
    register_file_tools(tool_registry, workspace, permissions=permissions)
    register_search_tools(tool_registry, workspace)
    register_command_tools(tool_registry, workspace, permissions, sandbox=sandbox)
    register_web_tools(tool_registry)
    register_browser_tools(tool_registry)

    # --- Git + Undo ---
    git = GitIntegration(workspace)
    undo_manager = UndoManager(workspace, git)

    # --- Context Manager ---
    context_manager = ContextManager(model=settings.model)

    # --- Conversation Logger ---
    conv_logger = ConversationLogger()
    conv_logger.log_system(f"Session started — model={settings.model}, workspace={workspace}")
    if project_info.language != "unknown":
        conv_logger.log_system(f"Project: {project_info.summary()}")

    # --- Mention Resolver ---
    mention_resolver = MentionResolver(workspace, git)

    # --- Phase 4: Codebase Indexer ---
    repo_map = RepoMap(workspace)
    # Skip indexing home dir in GUI mode — workspace picker will set a real project later
    if workspace != Path.home():
        try:
            indexed_count = repo_map.index()
        except Exception as exc:
            logger.warning("Repo indexing failed: %s", exc)
            indexed_count = 0
        if indexed_count > 0 and args.verbose:
            display.show_info(f"Indexed {indexed_count} files for repo map")
    else:
        indexed_count = 0

    # --- Phase 6: Memory Store ---
    memory_dir = Path.home() / ".unjess" / "memory"
    try:
        memory_store = MemoryStore(memory_dir)
        memory_store.ensure_loaded()
    except Exception as exc:
        logger.warning("Memory store init failed: %s", exc)
        memory_store = None

    # --- Phase 6: Knowledge Graph ---
    try:
        knowledge_graph = KnowledgeGraph()
        knowledge_graph.ensure_loaded()
    except Exception as exc:
        logger.warning("Knowledge graph init failed: %s", exc)
        knowledge_graph = None

    # --- Phase 6: Skills ---
    skill_engine: SkillEngine | None = None
    try:
        skill_roots = [
            workspace / ".agents" / "skills",
            Path.home() / ".unjess" / "skills",
        ]
        valid_roots = [r for r in skill_roots if r.is_dir()]
        if valid_roots:
            skill_engine = SkillEngine(skill_roots=valid_roots)
            count = skill_engine.discover()
            if count and (args.verbose or settings.verbosity == "verbose"):
                display.show_info(f"Discovered {count} skills")
    except Exception as exc:
        logger.warning("Skill engine init failed: %s", exc)

    # --- Phase 6: Embeddings + RAG (opt-in — enable with /settings enable_rag true) ---
    rag_engine = None
    if settings.enable_rag:
        try:
            embed_provider = get_embedding_provider(settings.api_keys)
            if embed_provider:
                vector_store = VectorStore(storage_dir=memory_dir, name="rag")
                vector_store.ensure_loaded()
                rag_engine = RAGEngine(vector_store, embed_provider)
                if args.verbose:
                    display.show_info(
                        f"RAG enabled ({vector_store.count} indexed docs)"
                    )
        except Exception as exc:
            logger.warning("RAG init failed: %s", exc)

    # --- Create Agent ---
    agent = Agent(
        settings=settings,
        router=router,
        tool_registry=tool_registry,
        display=display,
        permissions=permissions,
        context_manager=context_manager,
        conv_logger=conv_logger,
        undo_manager=undo_manager,
        mention_resolver=mention_resolver,
        repo_map=repo_map,
        memory_store=memory_store,
        rag_engine=rag_engine,
        knowledge_graph=knowledge_graph,
        skill_engine=skill_engine,
        input_handler=terminal_input,
    )

    # Register ask_question tool (needs agent reference)
    agent.register_ask_question()

    # --- Build command context for slash commands ---
    cmd_ctx = CommandContext(
        agent=agent,
        settings=settings,
        display=display,
        router=router,
        undo_manager=undo_manager,
        context_manager=context_manager,
        logger=conv_logger,
        repo_map=repo_map,
        memory_store=memory_store,
        input_handler=terminal_input,
    )

    # --- GUI mode ---
    if getattr(args, 'gui', False):
        try:
            from unjess.gui.app import launch as launch_gui
        except ImportError:
            display.show_error(
                "GUI dependencies not installed. Run: pip install unjess[gui]"
            )
            sys.exit(1)
        launch_gui(
            agent=agent,
            settings=settings,
            router=router,
            cmd_ctx=cmd_ctx,
            conv_logger=conv_logger,
            native=getattr(args, 'native', False),
            memory_store=memory_store,
        )
        # Auto-summarize after GUI session
        _auto_summarize(agent, router, settings, memory_store, rag_engine, knowledge_graph, display, conv_logger)
        return

    # --- One-shot mode ---
    if args.prompt:
        try:
            agent.run(args.prompt)
        except KeyboardInterrupt:
            console.print()
        except Exception as exc:
            display.show_error(f"Error: {exc}")
            sys.exit(1)
        # Show cost summary for one-shot mode
        if conv_logger.cost_tracker.llm_calls > 0:
            summary = conv_logger.cost_tracker.summary()
            if summary.get("is_free"):
                display.show_info(
                    f"Tokens: {summary['tokens_in']}→{summary['tokens_out']} | Cost: FREE"
                )
            else:
                display.show_info(
                    f"Tokens: {summary['tokens_in']}→{summary['tokens_out']} "
                    f"| Cost: ${summary['estimated_cost']:.4f}"
                )
        # Auto-summarize one-shot session
        _auto_summarize(agent, router, settings, memory_store, rag_engine, knowledge_graph, display, conv_logger)
        return

    # Auto-activate free mode if enabled in config (before banner so it shows correct model)
    if settings.free_mode_enabled:
        from unjess.free_mode import FreeMode
        free_mode = FreeMode(settings=settings, router=router, display=display)
        free_mode.activate()
        cmd_ctx._free_mode = free_mode  # type: ignore[attr-defined]
        agent._free_mode = free_mode  # type: ignore[attr-defined]

    # --- Interactive mode ---
    display.show_banner(
        model=settings.model,
        provider=router.active_provider_name,
        workspace=str(workspace),
    )

    # Show project info if detected
    if project_info.language != "unknown":
        display.show_info(f"Project: {project_info.summary()}")
    if git.is_git_repo:
        display.show_info(f"{git.context_summary()}")
    if indexed_count > 0:
        display.show_info(f"Repo map: {indexed_count} files indexed")

    _run_repl(agent, cmd_ctx, display)

    # Auto-summarize interactive session on exit
    _auto_summarize(agent, router, settings, memory_store, rag_engine, knowledge_graph, display, conv_logger)


def _auto_summarize(
    agent: Agent,
    router: ProviderRouter,
    settings: Settings,
    memory_store: "MemoryStore | None",
    rag_engine: "RAGEngine | None",
    knowledge_graph: "KnowledgeGraph | None",
    display: Display,
    conv_logger: "ConversationLogger | None" = None,
) -> None:
    """Generate and save a session summary on exit."""
    conversation = agent.get_conversation()
    if not conversation or len(conversation) < 2:
        return

    # Use the conversation logger's ID so the summary matches the transcript folder
    conv_id = conv_logger.conversation_id if conv_logger else ""

    try:
        summary = generate_summary(
            conversation=conversation,
            router=router,
            settings=settings,
            memory_store=memory_store,
            conversation_id=conv_id,
        )
        if summary:
            display.show_info(f"Session saved: {summary.title[:50]}")

            # Index summary in RAG
            if rag_engine and rag_engine.is_available:
                try:
                    rag_engine.index_summary(
                        summary_id=summary.conversation_id,
                        title=summary.title,
                        summary_text=summary.summary,
                        key_topics=summary.key_topics,
                    )
                    rag_engine.index_conversation(
                        summary.conversation_id, conversation
                    )
                except Exception as exc:
                    logger.debug("RAG indexing failed: %s", exc)

            # Save knowledge graph
            if knowledge_graph:
                try:
                    for topic in summary.key_topics:
                        knowledge_graph.record_concept(topic)
                    for f in summary.files_modified:
                        knowledge_graph.record_file_write(f)
                    knowledge_graph.save()
                except Exception as exc:
                    logger.debug("Knowledge graph save failed: %s", exc)
    except KeyboardInterrupt:
        pass  # User wants out NOW — skip summarization silently
    except Exception as exc:
        logger.debug("Auto-summary failed: %s", exc)


def gui_main() -> None:
    """Standalone GUI entry point — launches njss as a desktop app.

    Called by the ``njss-gui`` command. Equivalent to ``njss --gui --native``
    but doesn't require a terminal or command-line arguments.
    """
    sys.argv = [sys.argv[0], "--gui", "--native"]
    main()


if __name__ == "__main__":
    main()
