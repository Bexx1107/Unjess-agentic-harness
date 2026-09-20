<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_banner.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo_banner_light.png">
    <img src="assets/logo_banner.png" alt="Unjess — Agentic Harness" width="500">
  </picture>
</p>

<p align="center">
  <strong>A full-featured AI coding agent — like Cursor, Aider, or Claude Code, but yours.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#gui">GUI</a> •
  <a href="#providers">Providers</a> •
  <a href="#free-mode">Free Mode</a> •
  <a href="docs/USER_GUIDE.md">User Guide</a>
</p>

---

## What Is This?

Unjess (`njss`) is an open-source AI coding agent that reads your codebase, writes code, runs commands, and fixes bugs — powered by any LLM you choose. It works as a **CLI in your terminal** or a **GUI desktop app**.

```
> add auth middleware to the Express app
> fix the failing tests in test_parser.py  
> refactor the database layer to use connection pooling
> find and fix the memory leak
```

The agent will read your files, write changes (with diffs), run commands (with your approval), search your codebase, and iterate until it's done.

---

## Quick Start

### Install from source

```bash
git clone https://github.com/Bexx1107/unjess.git
cd unjess
pip install -e ".[all]"
```

### Run

```bash
# Terminal (CLI)
njss

# GUI in browser
njss --gui

# GUI native desktop window
njss-gui
```

### One-shot mode

```bash
njss "add error handling to server.py"
```

### Build standalone exe

```bash
pip install -e ".[package]"
build.bat
# Output: dist/njss/njss.exe
```

On first run, a setup wizard walks you through picking a provider, entering API keys, and choosing a model.

---

## Features

### Core Agent Loop
- **Streaming responses** with real-time token display
- **Tool calling** with parallel execution for read-only tools
- **Planning mode** — agent creates a plan before complex tasks
- **Stuck detection** — auto-detects repetitive tool loops
- **Context management** — auto-compaction when context gets full
- **Budget enforcement** — set a dollar cap per session

### Tools (19 built-in)

| Tool | What it does |
|------|-------------|
| `read_file` | Read files with line ranges |
| `write_file` | Create or overwrite files |
| `edit_file` | Find-and-replace with diff output |
| `list_dir` | Directory listing with sizes |
| `grep_search` | Regex search (ripgrep when available, Python fallback) |
| `run_command` | Shell execution with sandbox + approval |
| `search_web` | Web search via Google/DuckDuckGo |
| `read_url` | Fetch and convert URLs to markdown |
| `browser_*` | Navigate, screenshot, click, fill (Playwright) |
| `spawn_agent` | Spawn child agents for parallel tasks |
| `ask_question` | Interactive multi-choice questions |

### Safety
- **Command approval** — asks before running shell commands
- **Sandbox** — blocks dangerous commands (rm -rf /, fork bombs, etc.)
- **Secret scrubbing** — redacts API keys from command output
- **Undo** — `/undo` reverts file changes (git checkpoints or file snapshots)
- **Budget cap** — auto-stops when cost exceeds your limit
- **Workspace sandboxing** — file operations restricted to project directory

### Extensibility
- **MCP servers** — connect external tools via Model Context Protocol
- **Skills** — reusable instruction sets with trigger matching
- **Plugins** — bundled skills + agents in plugin directories
- **Subagents** — spawn child agents (research, coder, reviewer, tester)
- **Scheduler** — one-shot timers and recurring cron jobs
- **`@mentions`** — `@file.py`, `@dir/`, `@git diff`, `@web "query"`, `@url https://...`

---

## GUI

Launch with `njss --gui` (browser) or `njss-gui` (native window).

- Dark theme with customizable accent colors
- Real-time streaming with thinking display
- File browser sidebar
- Session stats (tokens, cost, model)
- Settings dialog with all config options
- Usage tracking with per-model cost breakdown
- Image upload (📎 button) and clipboard paste (Ctrl+V)
- Onboarding wizard for first-time setup

---

## Providers

| Provider | Models | Free tier? | Env var |
|----------|--------|:----------:|---------|
| **Google** | Gemini 3.5 Flash, 3.1 Pro | ✅ | `GOOGLE_API_KEY` |
| **Groq** | Llama 3.5 70B, Mixtral Large 3 | ✅ | `GROQ_API_KEY` |
| **Mistral** | Codestral v2, Mistral Large 3.5 | ✅ | `MISTRAL_API_KEY` |
| **OpenAI** | GPT-5.4, GPT-5.4-mini, o5-preview | ❌ | `OPENAI_API_KEY` |
| **Anthropic** | Claude Sonnet 5, Opus 4.8 | ❌ | `ANTHROPIC_API_KEY` |
| **xAI** | Grok 4.1 Fast, Grok 4.5 | ❌ | `XAI_API_KEY` |
| **OpenRouter** | Any model (free tier auto-pick) | ✅ | `OPENROUTER_API_KEY` |
| **Kimi** | Moonshot AI (kimi-k2.5, kimi-k3) | ❌ | `MOONSHOT_API_KEY` |
| **Qwen** | Alibaba DashScope (qwen-max, qwen-plus) | ❌ | `DASHSCOPE_API_KEY` |
| **Ollama** | Local & cloud models (Qwen, Llama, DeepSeek) | ✅ Local / ☁️ Cloud | Optional `OLLAMA_API_KEY` |

API keys are saved to `~/.unjess/keys.yaml` (0600 permissions). Enter once, never again.

---

## Free Mode

Use Unjess without paying anything. Free Mode auto-rotates between free-tier cloud models:

```bash
# Activate via slash command
/free

# Or pick "Free Mode" during setup
njss  # first run wizard
```

When one provider hits its rate limit, Unjess auto-switches to the next. Get free API keys from [ai.google.dev](https://ai.google.dev), [console.groq.com](https://console.groq.com), [console.mistral.ai](https://console.mistral.ai).

---

## CLI Flags

```
njss "prompt"               # One-shot mode
njss -m gemini-2.5-pro      # Use a specific model
njss -p google              # Force a specific provider
njss -w ./my-project        # Set workspace directory
njss --config path/to.yaml  # Custom config file
njss --verbose              # Debug output
njss --yes                  # Auto-approve all commands (⚠️ dangerous)
njss --gui                  # Launch web GUI
njss --native               # Native desktop window (with --gui)
njss --version              # Print version
```

## Slash Commands

| Command | What it does |
|---------|-------------|
| `/help` | Show all commands |
| `/model` | Interactive model picker |
| `/model gpt-4o` | Quick-switch model |
| `/free` | Activate/deactivate free mode |
| `/cost` | Show session cost and token usage |
| `/context` | Show context window usage |
| `/undo` | Undo last file changes |
| `/plan on/off` | Toggle plan-first mode |
| `/compact` | Compress conversation to save context |
| `/map` | Show codebase repo map |
| `/mcp` | List/connect MCP servers |
| `/skill` | List installed skills |
| `/learn <rule>` | Teach a persistent rule |
| `/settings` | View/change settings |
| `/setup` | Re-run setup wizard |

---

## Configuration

Config: `~/.unjess/config.yaml`

```yaml
model: gemini-2.5-flash
provider: google
free_mode_enabled: true
max_iterations: 25
confirm_commands: true
show_stats: true
show_diffs: true
max_cost_per_session: 0.0   # 0 = unlimited
```

Keys: `~/.unjess/keys.yaml`

---

## Project Structure

```
src/unjess/
├── cli.py                 # CLI entry point + REPL
├── agent.py               # Agent loop (streaming, tools, planning)
├── config.py              # Settings + YAML config
├── commands.py            # Slash command registry
├── display.py             # Terminal output (rich)
├── system_prompt.py       # Prompt builder
├── permissions.py         # Command approval system
├── sandbox.py             # Command blocklist + secret scrubbing
├── undo.py                # Git checkpoints + file snapshots
├── context_manager.py     # Token counting + auto-compaction
├── conversation_logger.py # JSONL transcripts + cost tracking
├── memory.py              # Session summaries + learned rules
├── mentions.py            # @file, @git, @web resolution
├── free_mode.py           # Free-tier model rotation
├── model_router.py        # Smart task routing
├── scheduler.py           # Timers + cron jobs
├── tools/
│   ├── file_tools.py      # read/write/edit/list
│   ├── search_tools.py    # grep (ripgrep)
│   ├── command_tools.py   # shell execution
│   ├── web_tools.py       # search + fetch URLs
│   ├── browser_tools.py   # Playwright automation
│   └── subagent_tools.py  # spawn/manage child agents
├── llm/
│   ├── base.py            # Abstract provider interface
│   ├── google_provider.py # Gemini (google-genai SDK)
│   ├── anthropic_provider.py # Claude (anthropic SDK)
│   ├── openai_compat.py   # OpenAI/Groq/Mistral/xAI/Ollama
│   ├── router.py          # Multi-provider routing + fallback
│   └── retry.py           # Exponential backoff
├── gui/
│   ├── app.py             # NiceGUI entry point
│   ├── pages/chat.py      # Main chat interface
│   ├── pages/onboarding.py # Setup wizard
│   ├── display.py         # GUI display adapter
│   └── components/        # UI components
├── mcp/                   # Model Context Protocol client
├── subagents/             # Child agent manager + types
├── skills/                # Skill discovery + engine
├── indexer/               # Repo map + symbol extraction
└── knowledge_graph.py     # Entity tracking
```

---

## Requirements

- Python 3.10+
- At least one API key (or Ollama for local models)

### Optional

- **ripgrep** (`rg`) — faster grep searches (falls back to Python grep)
- **Playwright** — browser automation (`pip install unjess[browser]`)
- **Pillow** — image support (`pip install unjess[vision]`)

---

## License

MIT

---

<p align="center">
  <sub>Built by <a href="https://github.com/Bexx1107">Bexx</a></sub>
</p>
