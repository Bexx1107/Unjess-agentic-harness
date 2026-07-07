# Unjess Project Rules

## Project Context
This is **Unjess (njss)**, a full-featured AI coding agent built in Python.
- CLI command: `njss`
- Package name: `unjess`
- Python 3.10+

## Architecture
- Full architecture blueprint: `docs/architecture.md` (46 systems across 5 parts)
- Phased build plan: `docs/build_plan.md` (6 phases with sub-phases)
- **READ BOTH DOCS** before making any changes to understand the full system design.

## Build Rules
- Follow the build plan phase by phase, sub-phase by sub-phase.
- Write complete files — don't leave TODOs or stubs unless explicitly noted.
- All file operations must be sandboxed to the workspace root.
- All LLM providers must implement the same abstract interface (`llm/base.py`).
- Use `rich` for terminal output. Use `prompt-toolkit` for input.
- Use `pyyaml` for config files.
- Keep the system prompt lean (~2,000 tokens) so local models can handle it.

## Code Style
- Python type hints on all function signatures.
- Docstrings on all classes and public methods.
- No wildcard imports.
- Use `pathlib.Path` instead of raw strings for file paths.
- Use `dataclasses` or simple classes, not Pydantic (keep dependencies minimal).
