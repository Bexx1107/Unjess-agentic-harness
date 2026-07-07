# Unjess (njss) — User Guide

> **Unjess** is a free, open-source AI coding agent that runs in your terminal. It reads your codebase, writes code, runs commands, and fixes bugs — powered by any LLM provider you choose.

---

## Quick Start

### Install
```bash
pip install unjess
```

### First Run
```bash
njss
```

On first run, a setup wizard walks you through:
1. **Pick a mode** — Free Mode (auto-rotate free models) or a specific provider
2. **Enter API keys** — paste them right in the wizard, they're saved for next time
3. **Pick a model** — or let Free Mode choose for you

That's it. You're coding.

### One-Shot Mode
```bash
njss "add error handling to server.py"
```
Runs once and exits — great for scripts and CI.

---

## CLI Flags

| Flag | What it does |
|---|---|
| `njss "prompt"` | One-shot mode — run a single task and exit |
| `njss -m gemini-2.5-pro` | Use a specific model |
| `njss -p google` | Force a specific provider |
| `njss -w ./my-project` | Set workspace directory |
| `njss --config path/to/config.yaml` | Custom config file |
| `njss --verbose` | Show debug output |
| `njss --yes` | Auto-approve all commands (⚠️ dangerous) |
| `njss --version` | Print version |

---

## Slash Commands

Type `/` in the prompt to see all commands. Here's every one:

### Core

| Command | Usage | What it does |
|---|---|---|
| `/help` | | Show all available commands |
| `/exit` or `/quit` | | Quit njss |
| `/clear` | | Clear conversation history (fresh start) |
| `/compact` | | Summarize and compress conversation to save context |

### Model & Provider

| Command | Usage | What it does |
|---|---|---|
| `/model` | | **Interactive picker** — lists all models from your provider's API, pick by number |
| `/model all` | | Lists models from ALL configured providers, pick by number |
| `/model gpt-4o` | `/model <name>` | Quick-switch to a specific model |
| `/free` | | Activate **Free Mode** — auto-rotates between free-tier models |
| `/free off` | | Deactivate free mode |
| `/free status` | | Show free-tier status (rate limits, remaining quota) |
| `/setup` | | Re-run the setup wizard (change provider, enter new API keys) |

### Context & Cost

| Command | Usage | What it does |
|---|---|---|
| `/context` | | Show context window usage (how full your conversation is) |
| `/cost` | | Show session cost, token usage, per-model breakdown |
| `/verbose` | | Enable verbose output |
| `/quiet` | | Enable quiet output |

### Code & Project

| Command | Usage | What it does |
|---|---|---|
| `/undo` | | Undo the last file change the agent made |
| `/undo list` | `/undo list` | Show undo history |
| `/undo 3` | `/undo <N>` | Undo the last N operations |
| `/map` | | Show the codebase repo map (file tree + symbols) |
| `/map --refresh` | | Re-index and show updated map |
| `/init` | | Generate a PROJECT.md context document for the agent |
| `/init --force` | | Regenerate even if one exists |

### Planning

| Command | Usage | What it does |
|---|---|---|
| `/plan` | | Show whether plan-first mode is on or off |
| `/plan on` | | Enable plan-first mode — agent creates a plan before coding |
| `/plan off` | | Disable plan-first mode — agent executes directly |

### Extensibility

| Command | Usage | What it does |
|---|---|---|
| `/skill` | | List all installed skills |
| `/skill django` | `/skill <name>` | Show details for a specific skill |
| `/mcp` | | List configured MCP servers |
| `/mcp connect db` | `/mcp connect <name>` | Connect to an MCP server |
| `/mcp disconnect db` | `/mcp disconnect <name>` | Disconnect from a server |
| `/mcp tools` | | List all available MCP tools |
| `/learn` | `/learn <rule>` | Teach the agent a persistent rule (remembered across sessions) |
| `/learn --forget 1` | `/learn --forget <id>` | Remove a learned rule |

### Other

| Command | Usage | What it does |
|---|---|---|
| `/schedule` | `/schedule 300 check tests` | Set a timer (seconds) with a reminder message |
| `/serve` | `/serve [host:port]` | Start the REST API server |
| `/settings` | | View all current settings |
| `/settings key value` | `/settings <key> <value>` | Change a setting |

---

## Free Mode

Free Mode lets you use njss **without paying anything**. It rotates between free-tier cloud models:

| Provider | Model | Limit |
|---|---|---|
| **Google** | Gemini 2.5 Flash | 1,500 requests/day |
| **Groq** | Llama 3.3 70B | 30 requests/min |
| **Mistral** | Codestral | Free for code tasks |

When one provider hits its rate limit, njss auto-switches to the next. Get free API keys from:
- [ai.google.dev](https://ai.google.dev)
- [console.groq.com](https://console.groq.com)
- [console.mistral.ai](https://console.mistral.ai)

**Activate:** `/free` or pick "Free Mode" during `/setup`

---

## Supported Providers

| Provider | Models | Free tier? | Key |
|---|---|---|---|
| **Google** | Gemini 2.5 Flash, Pro, etc. | ✅ 1,500 req/day | `GOOGLE_API_KEY` |
| **Groq** | Llama 3.3 70B, Mixtral, etc. | ✅ 30 req/min | `GROQ_API_KEY` |
| **Mistral** | Codestral, Nemo, etc. | ✅ Code models | `MISTRAL_API_KEY` |
| **OpenAI** | GPT-4o, GPT-4.1, o4-mini, etc. | ❌ Pay per token | `OPENAI_API_KEY` |
| **OpenAI OAuth** | Same as OpenAI | Uses your ChatGPT sub | Browser login |
| **Anthropic** | Claude Sonnet 4, Opus 4, etc. | ❌ Pay per token | `ANTHROPIC_API_KEY` |
| **Ollama** | Any local model | ✅ Always free | No key needed |

API keys are saved to `~/.unjess/keys.yaml` — enter once, never again.

---

## What Can It Do?

Just talk to it. Some examples:

```
> add auth middleware to the Express app
> fix the failing tests in test_parser.py
> refactor the database layer to use connection pooling
> explain how the payment flow works
> create a REST API for the todo app
> find and fix the memory leak
```

The agent will:
- 📖 **Read** your files to understand the codebase
- ✏️ **Write** code changes (with diffs shown)
- 🖥️ **Run** commands (with your approval)
- 🔍 **Search** your codebase for relevant code
- 🌐 **Browse** the web for documentation
- 🔄 **Iterate** — run tests, see errors, fix them

---

## Configuration

Config lives at `~/.unjess/config.yaml`:

```yaml
model: gemini-2.5-flash
provider: google
free_mode_enabled: true
max_iterations: 25
confirm_commands: true
show_stats: true
show_diffs: true
max_cost_per_session: 0.0   # 0 = unlimited, or set a dollar cap
```

API keys are stored separately at `~/.unjess/keys.yaml`.

---

## Ollama (Local Models)

1. Install from [ollama.com](https://ollama.com/download)
2. Pull a model: `ollama pull gemma3:12b-it-q4_K_M`
3. In njss: `/model gemma3:12b-it-q4_K_M`

Ollama is always available — no API key needed. Models run on your GPU.

---

## Safety

- **Command approval** — the agent asks before running any shell command
- **Sandbox** — dangerous commands (rm -rf /, fork bombs) are blocked outright
- **Secret scrubbing** — API keys in command output are redacted before entering the LLM context
- **Undo** — `/undo` reverts the last file change
- **Budget cap** — set `max_cost_per_session` to auto-stop when spending exceeds your limit

---

## Skills & MCP

**Skills** are reusable instruction sets. Drop `.json` files in `~/.unjess/skills/`:

```json
{
  "name": "django",
  "description": "Django conventions",
  "triggers": ["migration", "django"],
  "instructions": "Always use class-based views..."
}
```

**MCP** (Model Context Protocol) connects external tools. Configure in `~/.unjess/mcp.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    }
  }
}
```

Then: `/mcp connect github`

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit njss |
| `Tab` | Autocomplete slash commands |
| `↑ / ↓` | Command history |

---

*Built with ❤️ — MIT License*
