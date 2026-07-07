"""Slash command registry and handlers.

All slash commands are registered here and dispatched by the CLI.
Includes Phase 1-6 commands: /help, /model, /clear, /settings, /compact,
/context, /cost, /undo, /verbose, /quiet, /init, /map, /learn, /schedule, /serve.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

if TYPE_CHECKING:
    from unjess.agent import Agent
    from unjess.config import Settings
    from unjess.context_manager import ContextManager
    from unjess.conversation_logger import ConversationLogger
    from unjess.display import Display
    from unjess.llm.router import ProviderRouter
    from unjess.protocols import InputProtocol
    from unjess.undo import UndoManager


# ---------------------------------------------------------------------------
# Command context — passed to every handler
# ---------------------------------------------------------------------------

@dataclass
class CommandContext:
    """All dependencies a slash command handler might need."""

    agent: "Agent"
    settings: "Settings"
    display: "Display"
    router: "ProviderRouter"
    undo_manager: Optional["UndoManager"] = None
    context_manager: Optional["ContextManager"] = None
    logger: Optional["ConversationLogger"] = None
    repo_map: Optional[Any] = None
    memory_store: Optional[Any] = None
    input_handler: Optional["InputProtocol"] = None


# Handler signature: (ctx: CommandContext, args: str) -> bool
# Returns True to continue REPL, False to exit.
HandlerFn = Callable[[CommandContext, str], bool]


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

@dataclass
class SlashCommand:
    """A registered slash command."""

    name: str
    description: str
    handler: HandlerFn
    usage: str = ""


_COMMANDS: dict[str, SlashCommand] = {}


def get_all_commands() -> dict[str, SlashCommand]:
    """Return all registered slash commands.

    Used by the autocomplete system to build completion candidates.
    """
    return dict(_COMMANDS)


def register(name: str, description: str, handler: HandlerFn, usage: str = "") -> None:
    """Register a slash command."""
    _COMMANDS[name] = SlashCommand(
        name=name,
        description=description,
        handler=handler,
        usage=usage,
    )


def dispatch(command_str: str, ctx: CommandContext) -> bool:
    """Parse and dispatch a slash command.

    Args:
        command_str: The full command string (e.g. "/model gpt-4o").
        ctx: Command context with all dependencies.

    Returns:
        True to continue REPL, False to exit.
    """
    parts = command_str.strip().split(maxsplit=1)
    cmd_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    cmd = _COMMANDS.get(cmd_name)
    if cmd is None:
        ctx.display.show_error(f"Unknown command: {cmd_name}. Type /help for available commands.")
        return True

    return cmd.handler(ctx, args)


# ---------------------------------------------------------------------------
# Built-in command handlers
# ---------------------------------------------------------------------------

def _cmd_exit(ctx: CommandContext, args: str) -> bool:
    """Exit the REPL."""
    ctx.display.show_info("Goodbye! 👋")
    return False


def _cmd_help(ctx: CommandContext, args: str) -> bool:
    """Show all available commands."""
    table = Table(show_header=False, border_style="dim", expand=True, padding=(0, 1))
    table.add_column("Command", style="bold cyan", min_width=20, no_wrap=True)
    table.add_column("Description", style="dim", ratio=1)

    for cmd in sorted(_COMMANDS.values(), key=lambda c: c.name):
        usage = f"  {cmd.usage}" if cmd.usage else ""
        table.add_row(cmd.name + usage, cmd.description)

    panel = Panel(table, title="Commands", title_align="left", border_style="cyan", expand=True)
    ctx.display.console.print(panel)
    return True


def _cmd_model(ctx: CommandContext, args: str) -> bool:
    """Switch or display the active model. Use '/model' to pick from a list."""
    from rich.prompt import Prompt

    arg = args.strip()

    # Subcommands that should NOT be treated as model names
    _SUBCOMMANDS = {"all", "list", "help"}

    # --- /model <name> — quick switch ---
    if arg and arg.lower() not in _SUBCOMMANDS:
        old = ctx.settings.model
        ctx.settings.model = arg
        if ctx.context_manager:
            ctx.context_manager.model = arg
        ctx.display.show_info(f"Model: {old} → {arg}")
        return True

    # --- /model list — alias for /model all ---
    if arg.lower() == "list":
        arg = "all"

    # --- Check if free mode is active ---
    _in_free_mode = (
        hasattr(ctx, '_free_mode') and ctx._free_mode.is_active  # type: ignore[attr-defined]
    )

    # --- /model all — list providers, pick from any ---
    if arg.lower() == "all":
        if _in_free_mode:
            # In free mode — only show free models
            from unjess.free_mode import _FREE_TIERS
            ctx.display.show_info("Free mode active — showing free models only")
            flat_list: list[tuple[str, str]] = []
            available_providers = set(ctx.router.available_providers)
            for tier in sorted(_FREE_TIERS, key=lambda t: t.priority):
                if tier.provider in available_providers:
                    flat_list.append((tier.model, tier.provider))
            # Also add OpenRouter :free models from live API
            if "openrouter" in available_providers:
                live_or = ctx.router.list_models("openrouter")
                for m in live_or:
                    if (m, "openrouter") not in flat_list:
                        flat_list.append((m, "openrouter"))
            if not flat_list:
                ctx.display.show_error("No free models available. Run /setup to add free API keys.")
                return True
            _show_model_picker(ctx, flat_list, show_provider=True)
            ctx.display.console.print("  [dim]Tip: /free off then /model list to see all models[/dim]")
        else:
            ctx.display.show_info("Fetching models from all providers...")
            all_models = ctx.router.list_all_models()
            if not all_models:
                ctx.display.show_error("No models found from any provider.")
                return True

            # Build flat list with provider labels
            flat_list = []
            for provider_name, models in all_models.items():
                for m in models:
                    flat_list.append((m, provider_name))

            _show_model_picker(ctx, flat_list, show_provider=True)
        return True

    # --- /model (no args) — interactive pick from active provider ---
    provider = ctx.router.active_provider_name
    ctx.display.console.print(f"  Current:   [bold]{ctx.settings.model}[/bold] via [bold]{provider}[/bold]")
    ctx.display.console.print()

    if _in_free_mode:
        ctx.display.show_info("Free mode active — showing free models only")
        ctx.display.console.print("  [dim]Tip: /free off then /model to see all models[/dim]")
        from unjess.free_mode import _FREE_TIERS
        flat_list = [
            (tier.model, tier.provider)
            for tier in sorted(_FREE_TIERS, key=lambda t: t.priority)
            if tier.provider in set(ctx.router.available_providers)
        ]
        _show_model_picker(ctx, flat_list, show_provider=True)
        return True

    ctx.display.show_info(f"Fetching models from {provider}...")

    models = ctx.router.list_models()
    if not models:
        ctx.display.console.print("  [dim]Could not fetch model list from API.[/dim]")
        ctx.display.console.print(
            f"  [dim]Try: /model all | Providers: {', '.join(ctx.router.available_providers)}[/dim]"
        )
        return True

    flat_list = [(m, provider) for m in models]
    _show_model_picker(ctx, flat_list, show_provider=False)
    return True


def _show_model_picker(
    ctx: CommandContext,
    models: list[tuple[str, str]],
    show_provider: bool = False,
) -> None:
    """Show a numbered model list and prompt for selection.

    Args:
        ctx: Command context.
        models: List of (model_id, provider_name) tuples.
        show_provider: Whether to show provider column.
    """
    from rich.prompt import Prompt

    table = Table(
        show_header=True, border_style="dim", expand=False, padding=(0, 1),
    )
    table.add_column("#", style="bold", width=4)
    table.add_column("Model", style="bold green")
    if show_provider:
        table.add_column("Provider", style="cyan")
    table.add_column("", style="dim")

    active = ctx.settings.model
    for i, (model_id, provider_name) in enumerate(models, 1):
        marker = "← active" if model_id == active else ""
        if show_provider:
            table.add_row(str(i), model_id, provider_name, marker)
        else:
            table.add_row(str(i), model_id, marker)

    ctx.display.console.print(table)
    ctx.display.console.print()

    choice = Prompt.ask(
        "Pick a model (number or name, Enter to cancel)",
        default="",
    )

    if not choice.strip():
        ctx.display.console.print("  [dim]Cancelled.[/dim]")
        return

    # Try as number
    if choice.strip().isdigit():
        idx = int(choice.strip()) - 1
        if 0 <= idx < len(models):
            selected = models[idx][0]
        else:
            ctx.display.show_error(f"Invalid number. Pick 1-{len(models)}.")
            return
    else:
        selected = choice.strip()

    old = ctx.settings.model
    ctx.settings.model = selected
    if ctx.context_manager:
        ctx.context_manager.model = selected
    ctx.display.show_info(f"Model: {old} → {selected}")


def _cmd_clear(ctx: CommandContext, args: str) -> bool:
    """Clear conversation history."""
    ctx.agent.clear_history()
    ctx.display.show_info("Conversation history cleared.")
    return True


def _cmd_settings(ctx: CommandContext, args: str) -> bool:
    """View or change settings."""
    from unjess.settings_ui import show_settings, edit_setting, save_and_notify

    if not args:
        show_settings(ctx.settings, ctx.display.console)
    else:
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            ctx.display.show_error("Usage: /settings <key> <value>")
            return True

        key, value = parts
        success, message = edit_setting(ctx.settings, key, value)

        if success:
            ctx.display.show_info(message)
            result = save_and_notify(ctx.settings)
            ctx.display.show_info(result)
        else:
            ctx.display.show_error(message)

    return True


def _cmd_compact(ctx: CommandContext, args: str) -> bool:
    """Summarize and compress conversation history."""
    if ctx.context_manager is None:
        ctx.display.show_error("Context manager not available.")
        return True

    old_count = ctx.agent.conversation_length
    ctx.agent.conversation = ctx.context_manager.compact(ctx.agent.conversation)
    new_count = ctx.agent.conversation_length

    ctx.display.show_info(
        f"Compacted: {old_count} messages -> {new_count} messages"
    )
    return True


def _cmd_context(ctx: CommandContext, args: str) -> bool:
    """Show context window usage breakdown."""
    if ctx.context_manager is None:
        ctx.display.show_error("Context manager not available.")
        return True

    from unjess.system_prompt import build_system_prompt
    system_prompt = build_system_prompt(ctx.settings, ctx.agent.tool_registry)
    breakdown = ctx.context_manager.get_context_breakdown(system_prompt, ctx.agent.conversation)

    table = Table(show_header=False, border_style="dim", expand=False, padding=(0, 1))
    table.add_column("Section", style="bold")
    table.add_column("Tokens", style="green", justify="right")

    table.add_row("Context window", f"{breakdown['context_window']:,}")
    table.add_row("Response reserve", f"{breakdown['response_reserve']:,}")
    table.add_row("System prompt", f"{breakdown['system_prompt_tokens']:,}")
    table.add_row("Conversation", f"{breakdown['conversation_tokens']:,} ({breakdown['message_count']} msgs)")
    table.add_row("", "")
    table.add_row("[bold]Available[/bold]", f"[bold]{breakdown['available']:,}[/bold]")
    table.add_row("Utilization", f"{breakdown['utilization_pct']}%")

    panel = Panel(table, title="Context Usage", title_align="left", border_style="green", expand=True)
    ctx.display.console.print(panel)
    return True


def _cmd_cost(ctx: CommandContext, args: str) -> bool:
    """Show session cost and token usage."""
    if ctx.logger is None:
        ctx.display.show_error("Conversation logger not available.")
        return True

    summary = ctx.logger.cost_tracker.summary()

    table = Table(show_header=False, border_style="dim", expand=True, padding=(0, 1))
    table.add_column("Key", style="bold cyan", min_width=14, no_wrap=True)
    table.add_column("Value", ratio=1, style="green", justify="right")

    table.add_row("Tokens in", f"{summary['tokens_in']:,}")
    table.add_row("Tokens out", f"{summary['tokens_out']:,}")
    table.add_row("Total tokens", f"{summary['total_tokens']:,}")

    if summary.get("is_free"):
        table.add_row("Session cost", "[bold green]FREE[/bold green]")
    else:
        table.add_row("Session cost", f"${summary['estimated_cost']:.4f}")

    table.add_row("LLM calls", str(summary['llm_calls']))
    table.add_row("Tool calls", str(summary['tool_calls']))

    # Budget cap
    if ctx.settings.max_cost_per_session > 0:
        remaining = max(0, ctx.settings.max_cost_per_session - summary['estimated_cost'])
        table.add_row("", "")
        table.add_row("Budget cap", f"${ctx.settings.max_cost_per_session:.4f}")
        table.add_row("Remaining", f"${remaining:.4f}")

    panel = Panel(table, title="Session Cost", title_align="left", border_style="yellow", expand=True)
    ctx.display.console.print(panel)

    # Per-model breakdown (if multiple models used)
    per_model = summary.get("per_model", {})
    if len(per_model) > 1:
        model_table = Table(
            show_header=True, border_style="dim", expand=False, padding=(0, 1)
        )
        model_table.add_column("Model", style="bold cyan")
        model_table.add_column("Calls", justify="right")
        model_table.add_column("Tokens", justify="right")
        model_table.add_column("Cost", justify="right")

        for model_name, data in per_model.items():
            total_tok = data["tokens_in"] + data["tokens_out"]
            if data["is_free"]:
                cost_str = "[green]FREE[/green]"
            else:
                cost_str = f"${data['cost']:.4f}"
            model_table.add_row(
                model_name, str(data["calls"]),
                f"{total_tok:,}", cost_str,
            )

        ctx.display.console.print(
            Panel(model_table, title="Per-Model Breakdown",
                  title_align="left", border_style="dim", expand=False)
        )

    return True


def _cmd_undo(ctx: CommandContext, args: str) -> bool:
    """Undo the last agent operation."""
    if ctx.undo_manager is None:
        ctx.display.show_error("Undo manager not available.")
        return True

    if args.strip() == "list":
        entries = ctx.undo_manager.list_entries(n=10)
        if not entries:
            ctx.display.show_info("No checkpoints found.")
        else:
            ctx.display.console.print("\n[bold]Checkpoints:[/bold]")
            for entry in entries:
                ctx.display.console.print(f"  {entry.display()}")
            ctx.display.console.print()
        return True

    # Perform undo
    steps = 1
    if args.strip().isdigit():
        steps = int(args.strip())

    success, message = ctx.undo_manager.undo(steps=steps)
    if success:
        ctx.display.show_info(message)
    else:
        ctx.display.show_error(message)
    return True


def _cmd_verbose(ctx: CommandContext, args: str) -> bool:
    """Enable verbose output."""
    ctx.settings.verbosity = "verbose"
    ctx.display._verbose = True
    ctx.display.show_info("Verbose mode enabled.")
    return True


def _cmd_quiet(ctx: CommandContext, args: str) -> bool:
    """Enable quiet output."""
    ctx.settings.verbosity = "quiet"
    ctx.display._verbose = False
    ctx.display.show_info("Quiet mode enabled.")
    return True


def _cmd_init(ctx: CommandContext, args: str) -> bool:
    """Initialize project context document."""
    from pathlib import Path
    from unjess.project_init import init_project

    workspace = Path(ctx.settings.workspace)
    force = args.strip() == "--force"

    ctx.display.show_info("Scanning project structure...")
    context_path = init_project(workspace, force=force)
    ctx.display.show_info(f"Project context saved to {context_path}")
    return True


# ---------------------------------------------------------------------------
# Phase 4-6 commands
# ---------------------------------------------------------------------------

def _cmd_map(ctx: CommandContext, args: str) -> bool:
    """Show the codebase repo map."""
    if ctx.repo_map is None:
        ctx.display.show_error("Repo map not available.")
        return True

    # Refresh index
    if args.strip() == "--refresh":
        count = ctx.repo_map.index()
        ctx.display.show_info(f"Re-indexed {count} files")
        return True

    max_tokens = 2000
    if args.strip().isdigit():
        max_tokens = int(args.strip())

    map_text = ctx.repo_map.format(max_tokens=max_tokens)
    if map_text:
        ctx.display.console.print(Panel(
            map_text,
            title="Repo Map",
            title_align="left",
            border_style="blue",
            expand=False,
        ))
    else:
        ctx.display.show_info("No files indexed. Run /map --refresh")
    return True


def _cmd_learn(ctx: CommandContext, args: str) -> bool:
    """Teach the agent a persistent rule."""
    if ctx.memory_store is None:
        ctx.display.show_error("Memory store not available.")
        return True

    if not args.strip():
        # Show existing rules
        rules = ctx.memory_store.get_rules(workspace=ctx.settings.workspace)
        if not rules:
            ctx.display.show_info("No learned rules. Usage: /learn <rule>")
        else:
            ctx.display.console.print("\n[bold]Learned Rules:[/bold]")
            for rule in rules:
                scope = "(global)" if not rule.workspace else f"({rule.workspace})"
                ctx.display.console.print(f"  [{rule.id}] {rule.rule} {scope}")
            ctx.display.console.print()
        return True

    # Check for forget
    if args.strip().startswith("--forget "):
        rule_id = args.strip().split(maxsplit=1)[1]
        if ctx.memory_store.forget(rule_id):
            ctx.display.show_info(f"Forgot rule: {rule_id}")
        else:
            ctx.display.show_error(f"Rule not found: {rule_id}")
        return True

    # Learn new rule
    rule = ctx.memory_store.learn(
        rule=args.strip(),
        workspace=ctx.settings.workspace,
        source="user",
    )
    ctx.display.show_info(f"Learned: {rule.rule} ({rule.id})")
    return True


def _cmd_schedule(ctx: CommandContext, args: str) -> bool:
    """Set a one-shot timer or recurring schedule."""
    from unjess.scheduler import Scheduler

    if not args.strip():
        ctx.display.show_info(
            "Usage:\n"
            "  /schedule 60 Check build status    (one-shot, 60 seconds)\n"
            "  /schedule */5 * * * * Run tests    (cron, every 5 minutes)"
        )
        return True

    parts = args.strip().split(maxsplit=1)

    # Check if first arg is a number (timer) or cron expression
    if parts[0].isdigit():
        seconds = int(parts[0])
        prompt = parts[1] if len(parts) > 1 else "Timer fired"

        # Get or create scheduler (store on context)
        if not hasattr(ctx, '_scheduler'):
            ctx._scheduler = Scheduler()  # type: ignore[attr-defined]

        task_id = ctx._scheduler.schedule_timer(seconds, prompt)  # type: ignore[attr-defined]
        ctx.display.show_info(f"Timer set: {seconds}s ({task_id})")
    else:
        ctx.display.show_info(
            "Cron scheduling is available via the Scheduler API. "
            "For CLI use, try: /schedule <seconds> <message>"
        )

    return True


def _cmd_serve(ctx: CommandContext, args: str) -> bool:
    """Start the API server."""
    from unjess.api import AgentAPI

    api = AgentAPI()
    if not api.is_available:
        ctx.display.show_error(
            "API server requires aiohttp. Install with: pip install aiohttp"
        )
        return True

    api.configure(agent=ctx.agent, settings=ctx.settings)

    host = "127.0.0.1"
    port = 9757
    if args.strip():
        parts = args.strip().split(":")
        if len(parts) == 2:
            host = parts[0]
            port = int(parts[1])
        elif parts[0].isdigit():
            port = int(parts[0])

    ctx.display.show_info(f"Starting API server on http://{host}:{port}")
    ctx.display.show_info("Press Ctrl+C to stop")

    try:
        api.run()
    except KeyboardInterrupt:
        ctx.display.show_info("API server stopped")

    return True


def _cmd_plan(ctx: CommandContext, args: str) -> bool:
    """Toggle architect mode (plan-first) or show current status."""
    from unjess.model_router import ArchitectMode, ModelRouter

    # Access architect mode from the agent
    architect: ArchitectMode = ctx.agent._architect  # type: ignore[attr-defined]

    if not args.strip():
        status = "[green]enabled[/green]" if architect.enabled else "[red]disabled[/red]"
        ctx.display.console.print(f"  Plan-first mode: {status}")
        ctx.display.console.print(
            "  [dim]When enabled, complex tasks trigger a plan-first workflow.[/dim]"
        )
        ctx.display.console.print(
            "  [dim]Toggle with: /plan on | /plan off[/dim]"
        )
        return True

    arg = args.strip().lower()
    if arg in ("on", "enable", "yes", "true"):
        architect.enabled = True
        ctx.display.show_info("Plan-first mode enabled — complex tasks will require a plan.")
    elif arg in ("off", "disable", "no", "false"):
        architect.enabled = False
        ctx.display.show_info("Plan-first mode disabled — agent will execute directly.")
    else:
        ctx.display.show_error("Usage: /plan [on|off]")

    return True


def _cmd_free(ctx: CommandContext, args: str) -> bool:
    """Activate free-tier mode or show free-tier status."""
    # Lazy import to avoid circular imports
    from unjess.free_mode import FreeMode

    # Get or create FreeMode instance (store on context)
    if not hasattr(ctx, '_free_mode'):
        ctx._free_mode = FreeMode(  # type: ignore[attr-defined]
            settings=ctx.settings,
            router=ctx.router,
            display=ctx.display,
        )

    free_mode: FreeMode = ctx._free_mode  # type: ignore[attr-defined]

    arg = args.strip().lower()

    if arg in ("off", "disable"):
        free_mode.deactivate()
        return True

    if arg == "status":
        free_mode.show_status()
        return True

    # Default: activate free mode
    free_mode.activate()
    return True


def _cmd_setup(ctx: CommandContext, args: str) -> bool:
    """Re-run the first-run setup wizard."""
    from unjess.first_run import run_first_setup

    updated = run_first_setup(ctx.settings, ctx.display.console)

    # Apply ALL new settings to the live session
    ctx.settings.model = updated.model
    ctx.settings.provider = updated.provider
    ctx.settings.api_keys = updated.api_keys
    if hasattr(updated, "free_mode_enabled"):
        ctx.settings.free_mode_enabled = updated.free_mode_enabled

    # Reload providers so new API keys take effect immediately
    if hasattr(ctx.router, "reload_providers"):
        ctx.router.reload_providers()

    ctx.display.show_info(
        f"Settings updated — now using {updated.model} via {updated.provider} "
        f"({len(ctx.router.available_providers)} providers available)"
    )
    return True


def _cmd_skill(ctx: CommandContext, args: str) -> bool:
    """List, search, or show skill details."""
    from pathlib import Path
    from unjess.skills.engine import SkillEngine

    # Discover skills from standard roots
    workspace = Path(ctx.settings.workspace)
    roots = [
        workspace / ".agents" / "skills",
        Path.home() / ".unjess" / "skills",
    ]
    engine = SkillEngine(skill_roots=[r for r in roots if r.is_dir()])
    engine.discover()

    arg = args.strip()

    if not arg or arg == "list":
        # List all skills
        skills = engine.skills
        if not skills:
            ctx.display.show_info(
                "No skills found.\n"
                "  Add .json files to ~/.unjess/skills/\n\n"
                "  Example ~/.unjess/skills/django.json:\n"
                '  {\n'
                '    "name": "django",\n'
                '    "description": "Django conventions",\n'
                '    "triggers": ["migration"],\n'
                '    "instructions": "Always run makemigrations..."\n'
                '  }'
            )
            return True

        table = Table(
            show_header=True, border_style="dim", expand=False, padding=(0, 1),
        )
        table.add_column("Skill", style="bold cyan")
        table.add_column("Description", style="dim")
        table.add_column("Path", style="dim italic")

        for skill in sorted(skills, key=lambda s: s.name):
            table.add_row(skill.name, skill.description, str(skill.path))

        panel = Panel(
            table,
            title=f"Skills ({len(skills)})",
            title_align="left",
            border_style="magenta",
            expand=False,
        )
        ctx.display.console.print(panel)
        return True

    # Show details for a specific skill
    skill = engine.get_skill(arg)
    if skill is None:
        # Try trigger matching
        matches = engine.match(arg)
        if matches:
            skill = matches[0]
        else:
            ctx.display.show_error(f"Skill not found: {arg}")
            return True

    ctx.display.console.print(f"\n[bold cyan]{skill.name}[/bold cyan]")
    ctx.display.console.print(f"  Description: {skill.description}")
    ctx.display.console.print(f"  Path: {skill.path}")
    if skill.trigger_patterns:
        ctx.display.console.print(f"  Triggers: {', '.join(skill.trigger_patterns)}")
    ctx.display.console.print(f"  Scripts: {'yes' if skill.has_scripts else 'no'}")
    ctx.display.console.print(f"  Examples: {'yes' if skill.has_examples else 'no'}")
    ctx.display.console.print(f"  References: {'yes' if skill.has_references else 'no'}")

    # Show instructions preview
    instructions = engine.load(skill.name)
    if instructions:
        preview = instructions[:500]
        if len(instructions) > 500:
            preview += "\n..."
        ctx.display.console.print(f"\n[dim]{preview}[/dim]\n")

    return True


def _cmd_mcp(ctx: CommandContext, args: str) -> bool:
    """Manage MCP servers — list, connect, disconnect, call tools."""
    from pathlib import Path
    from unjess.mcp.server_manager import ServerManager

    # Use persistent ServerManager (store on context so tools stay registered)
    if not hasattr(ctx, '_mcp_manager'):
        mcp_json = Path.home() / ".unjess" / "mcp.json"
        config_yaml = Path.home() / ".unjess" / "config.yaml"

        if mcp_json.exists():
            ctx._mcp_manager = ServerManager(config_path=mcp_json)  # type: ignore[attr-defined]
        else:
            ctx._mcp_manager = ServerManager(config_path=config_yaml)  # type: ignore[attr-defined]

    mgr: ServerManager = ctx._mcp_manager  # type: ignore[attr-defined]

    arg = args.strip()
    parts = arg.split(maxsplit=1) if arg else []
    sub_cmd = parts[0].lower() if parts else ""
    sub_args = parts[1] if len(parts) > 1 else ""

    if not sub_cmd or sub_cmd == "list":
        # List configured servers
        servers = mgr.server_names
        connected = mgr.connected_servers

        if not servers:
            ctx.display.show_info(
                "No MCP servers configured.\n"
                "  Create ~/.unjess/mcp.json:\n\n"
                '  {\n'
                '    "mcpServers": {\n'
                '      "my-server": {\n'
                '        "command": "npx",\n'
                '        "args": ["-y", "my-mcp-server"]\n'
                '      }\n'
                '    }\n'
                '  }'
            )
            return True

        table = Table(
            show_header=True, border_style="dim", expand=False, padding=(0, 1),
        )
        table.add_column("Server", style="bold cyan")
        table.add_column("Status", style="dim")
        table.add_column("Tools", style="green", justify="right")

        for name in sorted(servers):
            status = "[green]connected[/green]" if name in connected else "[dim]disconnected[/dim]"
            tool_count = ""
            if name in connected:
                tools = [t for t in mgr.available_tools if t.name.startswith(f"{name}/")]
                tool_count = str(len(tools))
            table.add_row(name, status, tool_count)

        panel = Panel(
            table,
            title=f"MCP Servers ({len(servers)})",
            title_align="left",
            border_style="blue",
            expand=False,
        )
        ctx.display.console.print(panel)
        return True

    if sub_cmd == "connect":
        if not sub_args:
            ctx.display.show_error("Usage: /mcp connect <server-name>")
            return True
        server_name = sub_args.strip()
        ctx.display.show_info(f"Connecting to {server_name}...")
        success = mgr.connect(server_name)
        if success:
            # Register MCP tools into the agent's tool registry
            tool_reg = ctx.agent.tool_registry
            count = mgr.register_tools(tool_reg)
            ctx.display.show_info(
                f"Connected to {server_name} — {count} tools registered\n"
                f"  The model can now use these tools directly."
            )
        else:
            ctx.display.show_error(f"Failed to connect to {server_name}")
        return True

    if sub_cmd == "disconnect":
        if not sub_args:
            ctx.display.show_error("Usage: /mcp disconnect <server-name>")
            return True
        server_name = sub_args.strip()
        mgr.disconnect(server_name)
        ctx.display.show_info(f"Disconnected from {server_name}")
        return True

    if sub_cmd == "tools":
        # List all available tools
        tools = mgr.available_tools
        if not tools:
            ctx.display.show_info("No MCP tools available. Connect a server first.")
            return True

        table = Table(
            show_header=True, border_style="dim", expand=False, padding=(0, 1),
        )
        table.add_column("Tool", style="bold cyan")
        table.add_column("Description", style="dim", max_width=60)

        for tool in sorted(tools, key=lambda t: t.name):
            table.add_row(tool.name, tool.description[:60])

        panel = Panel(
            table,
            title=f"MCP Tools ({len(tools)})",
            title_align="left",
            border_style="blue",
            expand=False,
        )
        ctx.display.console.print(panel)
        return True

    if sub_cmd == "refresh":
        # Re-discover tools from all connected servers
        connected = mgr.connected_servers
        if not connected:
            ctx.display.show_info(
                "No MCP servers connected. Use /mcp connect <name> first."
            )
            return True

        ctx.display.show_info("Refreshing MCP tools from all connected servers...")
        tool_reg = ctx.agent.tool_registry
        results = mgr.refresh_tools(tool_registry=tool_reg)

        total = sum(results.values())
        details = ", ".join(f"{name}: {count}" for name, count in results.items())
        ctx.display.show_success(
            f"Refreshed {total} tools from {len(results)} server(s)\n"
            f"  {details}"
        )
        return True

    ctx.display.show_error(
        f"Unknown subcommand: {sub_cmd}\n"
        "  /mcp list                — list configured servers\n"
        "  /mcp connect <name>      — connect to a server\n"
        "  /mcp disconnect <name>   — disconnect from a server\n"
        "  /mcp tools               — list available tools\n"
        "  /mcp refresh             — re-discover tools from connected servers"
    )
    return True


# ---------------------------------------------------------------------------
# /provider — hot-swap provider mid-conversation
# ---------------------------------------------------------------------------

def _cmd_provider(ctx: CommandContext, args: str) -> bool:
    """Switch LLM provider mid-conversation, keeping full context."""
    from rich.prompt import Prompt

    arg = args.strip().lower()
    available = ctx.router.available_providers
    current = ctx.router.active_provider_name

    # --- Quick switch: /provider google ---
    if arg and arg in available:
        return _do_provider_switch(ctx, arg, current)

    # --- Show provider picker ---
    provider_info = {
        "google": ("Google Gemini", "gemini-2.5-flash"),
        "groq": ("Groq", "llama-3.3-70b-versatile"),
        "mistral": ("Mistral", "codestral-latest"),
        "openrouter": ("OpenRouter", "openrouter/free"),
        "cerebras": ("Cerebras", "zai-glm-4.7"),
        "xai": ("xAI (Grok)", "grok-4.1-fast"),
        "openai": ("OpenAI", "gpt-4o"),
        "anthropic": ("Anthropic", "claude-sonnet-4"),
        "ollama": ("Ollama", ""),
    }

    table = Table(
        show_header=True, border_style="dim", expand=True, padding=(0, 1),
    )
    table.add_column("#", style="bold", width=4)
    table.add_column("Provider", style="bold cyan")
    table.add_column("Default Model", style="green")
    table.add_column("Keys", justify="right")
    table.add_column("", style="dim")

    choices = []
    for i, prov in enumerate(available, 1):
        info = provider_info.get(prov, (prov.title(), ""))
        display_name, default_model = info
        pool_count = len(ctx.settings.api_key_pool.get(prov, []))
        key_str = str(pool_count) if pool_count > 1 else ("✓" if ctx.settings.api_keys.get(prov) else "—")
        marker = "← active" if prov == current else ""
        table.add_row(str(i), display_name, default_model or "(auto)", key_str, marker)
        choices.append(prov)

    panel = Panel(
        table, title="Available Providers",
        title_align="left", border_style="cyan", expand=True,
    )
    ctx.display.console.print(panel)

    choice = Prompt.ask(
        "Pick a provider (number or name, Enter to cancel)",
        default="",
    )

    if not choice.strip():
        ctx.display.console.print("  [dim]Cancelled.[/dim]")
        return True

    if choice.strip().isdigit():
        idx = int(choice.strip()) - 1
        if 0 <= idx < len(choices):
            target = choices[idx]
        else:
            ctx.display.show_error(f"Invalid number. Pick 1-{len(choices)}.")
            return True
    else:
        target = choice.strip().lower()
        if target not in available:
            ctx.display.show_error(f"Unknown provider: {target}. Available: {', '.join(available)}")
            return True

    if target == current:
        ctx.display.show_info(f"Already using {target}.")
        return True

    return _do_provider_switch(ctx, target, current)


def _do_provider_switch(ctx: "CommandContext", target: str, current: str) -> bool:
    """Execute the provider switch with context handoff."""
    from rich.prompt import Prompt

    models = ctx.router.list_models(target)
    default_models = {
        "google": "gemini-2.5-flash",
        "groq": "llama-3.3-70b-versatile",
        "mistral": "codestral-latest",
        "openrouter": "openrouter/free",
        "cerebras": "zai-glm-4.7",
        "xai": "grok-4.1-fast",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4",
    }

    new_model = default_models.get(target, "")
    if models:
        if new_model and new_model in models:
            pass
        else:
            new_model = models[0]

    if not new_model:
        if models:
            ctx.display.show_info(f"Models for {target}: {', '.join(models[:10])}")
            new_model = Prompt.ask("Pick a model", default=models[0])
        else:
            ctx.display.show_error(f"No models found for {target}.")
            return True

    old_model = ctx.settings.model
    old_provider = ctx.settings.provider

    ctx.settings.provider = target
    ctx.settings.model = new_model
    if ctx.context_manager:
        ctx.context_manager.model = new_model

    # Inject handoff note so new model has context
    ctx.agent._conversation.append({
        "role": "system",
        "content": (
            f"[System: Provider switched from {old_provider}/{old_model} to "
            f"{target}/{new_model}. Full conversation context is preserved. "
            f"Continue helping the user seamlessly.]"
        ),
    })

    ctx.display.show_info(
        f"🔄 Provider: {old_provider} → {target}\n"
        f"   Model:    {old_model} → {new_model}\n"
        f"   Context:  {len(ctx.agent._conversation)} messages carried over"
    )
    return True


# ---------------------------------------------------------------------------
# /goal — long-running task with extended autonomy
# ---------------------------------------------------------------------------

def _cmd_goal(ctx: CommandContext, args: str) -> bool:
    """Execute a task in goal mode — extended autonomy, won't stop early."""
    if not args.strip():
        ctx.display.show_error("Usage: /goal <task description>")
        return True

    ctx.display.show_info("🎯 Goal mode: max iterations raised, agent will persist until done.")
    ctx.agent._goal_mode = True
    try:
        ctx.agent.run(args.strip())
    finally:
        ctx.agent._goal_mode = False
    return True


# ---------------------------------------------------------------------------
# /browser — toggle browser tools
# ---------------------------------------------------------------------------

def _cmd_browser(ctx: CommandContext, args: str) -> bool:
    """Start browser mode or navigate to a URL."""
    if args.strip():
        # Direct navigation
        ctx.agent.run(f"Navigate the browser to {args.strip()} and describe what you see.")
    else:
        ctx.display.show_info(
            "Browser tools available: browser_navigate, browser_click, browser_type, "
            "browser_get_text, browser_screenshot, browser_eval_js, browser_close"
        )
        ctx.display.show_info("Use: /browser <url> to navigate, or ask the agent to use the browser.")
    return True


# ---------------------------------------------------------------------------
# /diff — show changes made this session
# ---------------------------------------------------------------------------

def _cmd_diff(ctx: CommandContext, args: str) -> bool:
    """Show diffs of all files modified this session."""
    if not ctx.undo_manager:
        ctx.display.show_error("Undo manager not available.")
        return True

    snapshots = ctx.undo_manager.list_checkpoints()
    if not snapshots:
        ctx.display.show_info("No changes recorded this session.")
        return True

    # Show the diff from the first checkpoint to now
    from pathlib import Path

    first = snapshots[0] if snapshots else None
    if first and hasattr(first, 'files'):
        changed = 0
        for filepath, old_content in first.files.items():
            fpath = Path(filepath)
            if fpath.exists():
                new_content = fpath.read_text(encoding="utf-8", errors="replace")
                if old_content != new_content:
                    ctx.display.show_diff(filepath, old_content, new_content)
                    changed += 1

        if changed == 0:
            ctx.display.show_info("No file changes detected.")
        else:
            ctx.display.show_info(f"{changed} file(s) modified this session.")
    else:
        ctx.display.show_info("No detailed snapshot data available. Use /undo list to see checkpoints.")

    return True


# ---------------------------------------------------------------------------
# /test — run test suite
# ---------------------------------------------------------------------------

def _cmd_test(ctx: CommandContext, args: str) -> bool:
    """Run the project's test suite."""
    test_cmd = args.strip() if args.strip() else ""
    if test_cmd:
        ctx.agent.run(f"Run this test command and report results: {test_cmd}")
    else:
        ctx.agent.run(
            "Detect the project's test framework and run the test suite. "
            "Report results concisely — show any failures."
        )
    return True


# ---------------------------------------------------------------------------
# /agents — manage subagents
# ---------------------------------------------------------------------------

def _cmd_agents(ctx: CommandContext, args: str) -> bool:
    """List or manage active subagents."""
    # Check if subagent_manager exists on agent
    manager = getattr(ctx.agent, '_subagent_manager', None)
    if not manager:
        ctx.display.show_info("Subagent system not initialized in this session.")
        return True

    parts = args.strip().split(maxsplit=1)
    action = parts[0] if parts else "list"

    if action == "list":
        agents = manager.list_all()
        if not agents:
            ctx.display.show_info("No subagents active.")
            return True

        table = Table(title="Active Subagents", show_lines=False)
        table.add_column("ID", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Status")
        table.add_column("Elapsed")

        for a in agents:
            elapsed = f"{a.elapsed:.1f}s" if hasattr(a, 'elapsed') else ""
            table.add_row(
                a.conversation_id[:12] + "...",
                a.role,
                a.status.value,
                elapsed,
            )
        ctx.display.console.print(table)

    elif action == "kill" and len(parts) > 1:
        target = parts[1].strip()
        if target == "all":
            count = manager.kill_all()
            ctx.display.show_info(f"Killed {count} subagent(s).")
        else:
            # Match by prefix
            killed = False
            for a in manager.list_all():
                if a.conversation_id.startswith(target):
                    manager.kill(a.conversation_id)
                    ctx.display.show_info(f"Killed {a.role} ({a.conversation_id[:12]}...)")
                    killed = True
                    break
            if not killed:
                ctx.display.show_error(f"No subagent found matching: {target}")

    else:
        ctx.display.show_info("Usage: /agents [list|kill <id|all>]")

    return True


# ---------------------------------------------------------------------------
# /budget — quick cost limit setter
# ---------------------------------------------------------------------------

def _cmd_budget(ctx: CommandContext, args: str) -> bool:
    """Set or view the session cost budget."""
    if not args.strip():
        current = ctx.settings.max_cost_per_session
        if current > 0:
            spent = 0.0
            if ctx.logger:
                spent = ctx.logger.cost_tracker.total_cost
            ctx.display.show_info(f"Budget: ${current:.2f} | Spent: ${spent:.4f} | Remaining: ${current - spent:.4f}")
        else:
            ctx.display.show_info("No budget limit set. Use: /budget <amount>")
        return True

    try:
        amount = float(args.strip().replace("$", ""))
        ctx.settings.max_cost_per_session = amount
        ctx.display.show_info(f"Session budget set to ${amount:.2f}")
    except ValueError:
        ctx.display.show_error(f"Invalid amount: {args.strip()}")

    return True


# ---------------------------------------------------------------------------
# Register all built-in commands
# ---------------------------------------------------------------------------

_registered = False


def register_builtin_commands() -> None:
    """Register all built-in slash commands."""
    global _registered
    if _registered:
        return
    _registered = True

    # Phase 1-2: Core
    register("/exit",     "Quit njss",                            _cmd_exit)
    register("/quit",     "Quit njss",                            _cmd_exit)
    register("/help",     "Show available commands",              _cmd_help)
    register("/model",    "View or switch model",                 _cmd_model,    usage="<name>")
    register("/clear",    "Clear conversation history",           _cmd_clear)
    register("/settings", "View or change settings",             _cmd_settings, usage="[key] [value]")
    register("/compact",  "Summarize and compress conversation", _cmd_compact)
    register("/context",  "Show context window usage",           _cmd_context)
    register("/cost",     "Show session cost and token usage",   _cmd_cost)
    register("/undo",     "Undo last agent operation",           _cmd_undo,     usage="[list|N]")
    register("/verbose",  "Enable verbose output",               _cmd_verbose)
    register("/quiet",    "Enable quiet output",                 _cmd_quiet)
    register("/init",     "Generate project context document",   _cmd_init,     usage="[--force]")
    register("/setup",    "Re-run provider and model setup",     _cmd_setup)

    # Phase 3: Extensibility
    register("/skill",    "List or inspect skills",              _cmd_skill,    usage="[name]")
    register("/mcp",      "Manage MCP servers",                  _cmd_mcp,      usage="[list|connect|disconnect|tools|refresh]")

    # Phase 4-6: Intelligence + Platform
    register("/map",      "Show codebase repo map",              _cmd_map,      usage="[--refresh|tokens]")
    register("/learn",    "Teach a persistent rule",             _cmd_learn,    usage="<rule>|--forget <id>")
    register("/schedule", "Set a timer or schedule",             _cmd_schedule, usage="<seconds> <msg>")
    register("/serve",    "Start the API server",                _cmd_serve,    usage="[host:port]")
    register("/plan",     "Toggle plan-first mode",              _cmd_plan,     usage="[on|off]")
    register("/free",     "Activate free-tier mode",             _cmd_free,     usage="[off|status]")
    register("/provider", "Switch provider mid-conversation",    _cmd_provider, usage="[name]")

    # New: Autonomy + Browser + Diagnostics
    register("/goal",     "Run task with extended autonomy",     _cmd_goal,     usage="<task>")
    register("/browser",  "Browser mode — navigate or browse",   _cmd_browser,  usage="[url]")
    register("/diff",     "Show files changed this session",     _cmd_diff)
    register("/test",     "Run the project's test suite",        _cmd_test,     usage="[command]")
    register("/agents",   "List or manage subagents",            _cmd_agents,   usage="[list|kill <id|all>]")
    register("/budget",   "Set session cost budget",             _cmd_budget,   usage="[amount]")


# Auto-register on import
register_builtin_commands()

