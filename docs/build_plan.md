# Unjess (njss) — Build Plan

All 46 systems from the architecture doc, broken into 6 phases with sub-phases. Phase 1 is file-by-file ready to execute now. Phases 2-6 are detailed enough to start in future conversations.

---

## Phase 1: Core Agent (This Conversation)

> **Goal**: A working `njss` command that can chat, read/write/edit files, run commands, stream responses, and work with OpenAI, Anthropic, Google, or Ollama.

### Sub-phase 1a: Project Scaffold

| File | Lines | What |
|---|---|---|
| `pyproject.toml` | 35 | Dependencies, `njss` script entry, metadata |
| `src/unjess/__init__.py` | 5 | Package init, version |
| `src/unjess/__main__.py` | 5 | `python -m unjess` support |

**Dependencies to include**: `openai`, `anthropic`, `google-genai`, `httpx`, `rich`, `prompt-toolkit`, `pyyaml`, `tiktoken`

---

### Sub-phase 1b: Config System

| File | Lines | What |
|---|---|---|
| `src/unjess/config.py` | 120 | Settings dataclass, YAML load/save, CLI arg overrides, defaults |

**Covers**:
- Model / provider selection
- API keys (from env vars or config file)
- Workspace path
- Fallback chain
- Display settings (verbosity, show_diffs, show_cost)
- Behavior settings (max_iterations, confirm_commands)
- Budget limits
- Config file location: `~/.unjess/config.yaml`
- Auto-create default config on first run

---

### Sub-phase 1c: LLM Providers

| File | Lines | What |
|---|---|---|
| `src/unjess/llm/__init__.py` | 5 | Package init |
| `src/unjess/llm/base.py` | 80 | Abstract `LLMProvider` class, `LLMResponse` dataclass, `ToolCall` dataclass |
| `src/unjess/llm/openai_compat.py` | 150 | OpenAI-compatible provider (also used by Ollama). Streaming, function calling, error handling |
| `src/unjess/llm/anthropic_provider.py` | 150 | Anthropic/Claude provider. Different message format, content blocks, streaming |
| `src/unjess/llm/google_provider.py` | 140 | Google Gemini provider. google-genai SDK, function declarations |
| `src/unjess/llm/router.py` | 100 | Provider registry, model→provider routing, fallback chain logic |

**Key design decisions**:
- `openai_compat.py` handles both OpenAI API and Ollama (just different `base_url`)
- All providers implement the same `chat(messages, tools, stream) -> LLMResponse` interface
- Router picks provider based on model name or explicit provider setting
- Fallback: if primary fails, try next in chain
- Streaming returns an async iterator of chunks

**Provider detection logic in router**:
```
model starts with "gpt-" or "o1" or "o3" → openai
model starts with "claude-" → anthropic
model starts with "gemini-" → google
model contains ":" (e.g., "qwen2.5:14b") → ollama
explicit provider setting → override above
```

---

### Sub-phase 1d: Core Tools

| File | Lines | What |
|---|---|---|
| `src/unjess/tools/__init__.py` | 80 | `ToolRegistry` class — register, list, execute, JSON schema generation |
| `src/unjess/tools/file_tools.py` | 200 | `read_file`, `write_file`, `edit_file`, `list_dir` |
| `src/unjess/tools/search_tools.py` | 60 | `grep_search` (shells out to `rg` / ripgrep) |
| `src/unjess/tools/command_tools.py` | 80 | `run_command` (subprocess, timeout, approval gate) |

**Tool schemas** (JSON schema format for function calling):

- **read_file**: `(path: str, start_line?: int, end_line?: int) -> str`
- **write_file**: `(path: str, content: str, overwrite?: bool) -> str`
- **edit_file**: `(path: str, target: str, replacement: str) -> str`
  - Exact string match of `target` in file, replace with `replacement`
  - Show diff after edit
- **list_dir**: `(path: str) -> str`
  - Show files/dirs with sizes, ignore .git/node_modules etc.
- **grep_search**: `(query: str, path: str, case_insensitive?: bool, regex?: bool) -> str`
  - Shell out to `rg` (ripgrep), cap at 50 results
- **run_command**: `(command: str, cwd?: str) -> str`
  - Requires user approval (unless auto-approved)
  - Timeout: 30s default
  - Capture stdout + stderr

**All paths resolved relative to workspace root. Security: reject paths outside workspace.**

---

### Sub-phase 1e: System Prompt

| File | Lines | What |
|---|---|---|
| `src/unjess/system_prompt.py` | 80 | Template builder, dynamic sections |

**Initial system prompt blocks** (simplified for Phase 1):
1. Identity (who you are)
2. User info (OS, workspace, model)
3. Available tools (auto-generated from registry)
4. Guidelines (keep responses concise, use markdown)
5. Communication style

Total: ~1,500-2,000 tokens. Lean enough for small local models.

---

### Sub-phase 1f: Agent Loop

| File | Lines | What |
|---|---|---|
| `src/unjess/agent.py` | 200 | Main agent loop — think/act/observe cycle |

**The loop**:
```
1. Build system prompt
2. Send messages + tools to LLM (streaming)
3. If response has tool calls:
   a. For each tool call:
      - Validate args
      - Check permissions (ask user if needed)
      - Execute tool
      - Collect result
   b. Append tool results to messages
   c. Go to step 2 (let model see results)
4. If response is text only:
   a. Display response
   b. Break (wait for next user input)
```

**Includes**:
- Max iteration guard (default 50)
- Error recovery (malformed tool calls → retry message)
- Token counting (basic, for context tracking)
- Conversation history management

---

### Sub-phase 1g: Display Layer

| File | Lines | What |
|---|---|---|
| `src/unjess/display.py` | 150 | Streaming output, diff display, spinners, formatting |

**Features**:
- `stream_text(chunks)` — print tokens as they arrive
- `show_diff(path, old, new)` — syntax-highlighted unified diff
- `show_tool_call(name, args)` — "🔧 Calling read_file(src/app.py)..."
- `show_tool_result(name, result)` — formatted, truncated if long
- `show_error(message)` — red error display
- `show_info(message)` — dim info text
- `show_stats(tokens_in, tokens_out, cost)` — after each response
- Uses `rich` library for colors, panels, markdown rendering

---

### Sub-phase 1h: CLI + REPL

| File | Lines | What |
|---|---|---|
| `src/unjess/cli.py` | 120 | Arg parsing, banner, interactive REPL loop |

**CLI arguments**:
```
njss                           # interactive mode
njss "fix the bug"             # one-shot mode
njss -m gpt-4o                 # specify model
njss -p ollama                 # specify provider
njss -w ./my-project           # specify workspace
njss --config ~/.unjess/config.yaml
njss --version
```

**REPL loop**:
```
while True:
    user_input = prompt("> ")  # using prompt-toolkit (history, multiline)
    
    if user_input.startswith("/"):
        handle_slash_command(user_input)
    elif user_input in ("exit", "quit"):
        break
    else:
        agent.run(user_input)
```

**Slash commands (Phase 1 — basic set)**:
- `/help` — show available commands
- `/model <name>` — switch model
- `/clear` — clear conversation history
- `/exit` — quit

---

### Sub-phase 1i: Permissions

| File | Lines | What |
|---|---|---|
| `src/unjess/permissions.py` | 80 | Approval manager, auto-approve rules |

**Rules**:
- `read_file`, `list_dir`, `grep_search` → always auto-approve
- `write_file`, `edit_file` → auto-approve within workspace
- `run_command` → **always ask** (unless user says "Always")
- Show command preview, accept Y/N/A (yes/no/always)

---

### Sub-phase 1j: Error Handling

Built into `agent.py` and `llm/*.py`, not a separate file.

**Handles**:
- Malformed JSON tool calls → inject error message, let model retry (max 3)
- API rate limits → exponential backoff (1s, 2s, 4s)
- API timeout → retry once, then fail gracefully
- Tool execution errors → format error for model, let it self-correct
- File not found → tell model, let it try another path
- Context overflow → truncate oldest messages

---

### Sub-phase 1k: Test & Fix

- `pip install -e .` in the project directory
- Run `njss --version` to verify CLI works
- Run `njss` to test interactive mode
- Test with available provider
- Fix whatever broke

---

### Phase 1 Totals

| Sub-phase | Files | Lines |
|---|---|---|
| 1a: Scaffold | 3 | 45 |
| 1b: Config | 1 | 120 |
| 1c: LLM Providers | 6 | 625 |
| 1d: Core Tools | 4 | 420 |
| 1e: System Prompt | 1 | 80 |
| 1f: Agent Loop | 1 | 200 |
| 1g: Display | 1 | 150 |
| 1h: CLI + REPL | 1 | 120 |
| 1i: Permissions | 1 | 80 |
| 1j: Error Handling | 0 | (built into agent + llm) |
| 1k: Test & Fix | 0 | — |
| **Total** | **19 files** | **~1,840 lines** |

---
---

## Phase 2: Polish & UX (Next Conversation)

> **Goal**: Make it pleasant to use daily. Settings, logging, workspace awareness, git, undo.

### Sub-phase 2a: Expanded Slash Commands
- `/settings` — interactive settings panel
- `/model <name>` — enhanced with provider auto-detection
- `/compact` — summarize and compress conversation
- `/context` — show context usage breakdown
- `/cost` — show token/cost stats
- `/undo` — revert last agent change
- `/verbose` / `/quiet` — change verbosity
- **~200 lines**, modify `cli.py` + new `commands.py`

### Sub-phase 2b: Conversation Logging
- JSONL transcript writer (append-only)
- Log every step: user input, model response, tool calls, results
- Store in `~/.unjess/conversations/{id}/transcript.jsonl`
- **~200 lines**, new `logging.py`

### Sub-phase 2c: Settings UI
- Full `/settings` display (the pretty box from architecture doc)
- Live setting changes persisted to YAML
- Settings notification bus (other systems react to changes)
- **~300 lines**, new `settings_ui.py`

### Sub-phase 2d: Workspace Management
- Auto-detect project root (.git, package.json, Cargo.toml, etc.)
- `.agentignore` file support
- Project type detection ("This is a Next.js TypeScript project")
- **~250 lines**, new `workspace.py`

### Sub-phase 2e: Git Integration
- `git status`, `git diff`, `git log` helpers
- Git-aware context (current branch, uncommitted changes)
- Auto-checkpoint before agent edits (silent commits)
- **~250 lines**, new `git_integration.py`

### Sub-phase 2f: Undo / Rollback
- Git-based undo (revert to checkpoint)
- `/undo` command
- List checkpoints with `/undo list`
- **~200 lines**, new `undo.py`

### Sub-phase 2g: Basic Context Management
- Token counting per section
- Auto-truncate oldest messages when approaching limit
- `/compact` implementation (summarize old messages via cheap model)
- Context budget display
- **~300 lines**, modify `agent.py` + new `context_manager.py`

### Sub-phase 2h: Loop Control
- Max iterations per message (configurable)
- Consecutive error detection (stop after 3 in a row)
- Stuck detection (repeating same action)
- **~150 lines**, modify `agent.py`

### Phase 2 Totals: ~1,850 lines, ~8 new files

---

## Phase 3: Extensibility (Future Conversation)

> **Goal**: MCP, skills, plugins, @ mentions — the agent becomes extensible.

### Sub-phase 3a: MCP Client
- JSON-RPC client (stdio + HTTP transport)
- Server discovery from config
- Tool schema loading (lazy + eager)
- Tool proxying (expose MCP tools to agent)
- Server lifecycle management
- **~750 lines**, new `mcp/` directory (5 files)

### Sub-phase 3b: Skills System
- Skill discovery (scan directories, parse SKILL.md frontmatter)
- Trigger matching (match user intent to skill descriptions)
- Skill loading (inject SKILL.md instructions into context)
- **~330 lines**, new `skills/` directory (3 files)

### Sub-phase 3c: Plugin Architecture
- Plugin discovery (scan plugin directories)
- plugin.json parsing
- Bundle skills + agents from plugins
- **~200 lines**, new `plugins.py`

### Sub-phase 3d: @ Mentions
- Parse `@file.py`, `@web "query"`, `@git diff` syntax
- Resolve mentions to context injections
- File, directory, symbol, web, git resolvers
- **~340 lines**, new `mentions.py`

### Sub-phase 3e: Project Init (/init)
- Scan project structure
- Detect language, framework, build system, test framework
- Generate `.unjess/project_context.md`
- Load on future conversations
- **~300 lines**, new `project_init.py`

### Sub-phase 3f: Diff Display Enhancement
- Syntax-highlighted diffs (per language)
- File created / deleted display
- Side-by-side mode option
- **~130 lines**, modify `display.py`

### Phase 3 Totals: ~2,050 lines, ~12 new files

---

## Phase 4: Intelligence (Future Conversation)

> **Goal**: The agent gets smarter — repo awareness, testing loops, model routing.

### Sub-phase 4a: Codebase Indexing / Repo Map
- File scanner with ignore patterns
- Symbol extractor (regex-based initially, tree-sitter later)
- Repo map formatter (compressed to fit context)
- Incremental refresh on file changes
- **~500 lines**, new `indexer/` directory (4 files)

### Sub-phase 4b: Dual-Model / Router
- Complexity assessment (heuristic-based)
- Task classifier (edit, explain, plan, debug)
- Route to cheap vs expensive model
- Architect mode (plan with expensive, execute with cheap)
- **~400 lines**, modify `llm/router.py` + new `model_router.py`

### Sub-phase 4c: Testing Loop
- Auto-detect test command (pytest, npm test, cargo test, etc.)
- Edit → Run Tests → Fix Failures → Repeat cycle
- Test output parsers (per framework)
- Max fix attempts (default 5)
- **~450 lines**, new `testing.py`

### Sub-phase 4d: Lint Feedback Loop
- Run linter after edits
- Parse lint output into structured errors
- Feed back to model for self-correction
- **~200 lines**, new `lint_feedback.py`

### Sub-phase 4e: Advanced Context Management
- 5 context modes (Full, Lean, Auto-Compact, Manual, Adaptive)
- Token budget system with priorities
- Relevance scoring for files
- Smart section inclusion based on available space
- **~600 lines**, major rewrite of `context_manager.py`

### Phase 4 Totals: ~2,150 lines, ~8 new files

---

## Phase 5: Multi-Agent (Future Conversation)

> **Goal**: Spawn child agents, orchestrate complex tasks.

### Sub-phase 5a: Subagent System (Base)
- Define agent types with custom prompts and tool sets
- Invoke agents with a task, get conversation ID
- Send/receive messages between agents
- Agent lifecycle management (list, kill)
- **~650 lines**, new `subagents/` directory (4 files)

### Sub-phase 5b: Agent Types & Patterns
- Pre-defined types: research, coder, reviewer, tester, self
- Fan-out (parallel research)
- Pipeline (plan → code → review → test)
- Supervisor (managed team)
- **~500 lines**, extend `subagents/`

### Sub-phase 5c: Workspace Isolation
- Inherit / Branch / Share / Sandbox strategies
- Git worktree integration for branched workspaces
- Merge results back
- **~300 lines**, new `workspace_isolation.py`

### Sub-phase 5d: Task Orchestration Engine
- Workflow definition (steps with dependencies)
- Workflow executor (parallel where possible)
- Conditional branching
- Retry logic
- Progress visualization
- **~780 lines**, new `orchestration/` directory (4 files)

### Sub-phase 5e: Result Aggregation
- Merge, best-pick, synthesize, vote strategies
- **~100 lines**, new `aggregation.py`

### Phase 5 Totals: ~2,330 lines, ~10 new files

---

## Phase 6: Platform (Future Conversation)

> **Goal**: API layer, browser, OAuth, scheduling — become a platform.

### Sub-phase 6a: Extension / API Layer
- FastAPI HTTP server
- WebSocket streaming
- REST endpoints (chat, approve, config, status)
- CLI ↔ API client abstraction
- **~700 lines**, new `api/` directory (4 files)

### Sub-phase 6b: Browser Automation
- Playwright integration
- navigate, click, type, screenshot, get_text
- Can be exposed as MCP server or built-in tools
- **~400 lines**, new `browser/` directory (3 files)

### Sub-phase 6c: Image Understanding
- Base64 encode images for multimodal models
- Screenshot tool
- Image resize for context budget
- **~150 lines**, new `vision.py`

### Sub-phase 6d: OAuth Provider (ChatGPT Plus)
- Browser-based OAuth login flow
- Session token capture and storage
- Internal ChatGPT API client
- Token refresh logic
- **~500 lines**, new `llm/oauth_provider.py` + `llm/session_manager.py`

### Sub-phase 6e: Scheduling / Timers
- One-shot timers (remind after N seconds)
- Cron-based recurring tasks
- Background task integration
- **~200 lines**, new `scheduler.py`

### Sub-phase 6f: Conversation Memory (Cross-Session)
- Save conversation summaries
- Load recent summaries into context
- `/learn` command (persist behavioral rules)
- Optional: embedding-based search of past conversations
- **~300 lines**, new `memory.py`

### Phase 6 Totals: ~2,250 lines, ~10 new files

---
---

## Grand Summary

| Phase | Focus | Lines | Files | Session |
|---|---|---|---|---|
| **Phase 1** | Core Agent | ~1,840 | 19 | **This conversation** |
| **Phase 2** | Polish & UX | ~1,850 | 8 | Next conversation |
| **Phase 3** | Extensibility | ~2,050 | 12 | Future |
| **Phase 4** | Intelligence | ~2,150 | 8 | Future |
| **Phase 5** | Multi-Agent | ~2,330 | 10 | Future |
| **Phase 6** | Platform | ~2,250 | 10 | Future |
| **TOTAL** | | **~12,470** | **67 files** | |

> [!NOTE]
> The total is lower than the architecture doc's 15,000-21,000 estimate because:
> - OAuth piggyback moved to Phase 6 (complex, fragile)
> - Some systems merged (error handling built into agent loop, not separate)
> - System prompt templates are markdown files, not counted as code lines
> - Estimates tightened after thinking through actual implementations

> [!TIP]
> **For future conversations**: Reference the [architecture doc](docs/architecture.md) and this build plan. Tell the agent: "I'm building Unjess. Here's the architecture doc and build plan. Execute Phase 2." The architecture doc has all the design details; this plan has the file-by-file execution order.
