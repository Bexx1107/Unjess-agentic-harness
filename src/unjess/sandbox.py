"""Sandbox — command blocklist and secret detection for safe execution."""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Patterns that are ALWAYS blocked, no user override
_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"rm\s+-[a-z]*r[a-z]*\s+/(?:\s|$|;|\||&)",  # rm -rf / (combined flags)
        r"rm\s+-[a-z]*r[a-z]*\s+/\*",                # rm -rf /*
        r"rm\s+-[a-z]*r[a-z]*\s+~(?:\s|$|;|\||&)",   # rm -rf ~
        r"rm\s+-[a-z]*r[a-z]*\s+~/",                  # rm -rf ~/
        r"rm\s+.*--recursive.*\s+/",                   # rm --recursive /
        r"rm\s+-\w+\s+-\w+\s+/",                      # rm -f -r / (separate flags)
        r"find\s+/\s+.*-delete",                       # find / -delete
        r"find\s+/\s+.*-exec\s+rm",                    # find / -exec rm
        r"mkfs\.",                                     # format filesystem
        r"dd\s+if=.*of=/dev/",                        # overwrite disk
        r":\(\)\{\s*:\|:\s*\&\s*\}\s*;",                # fork bomb
        r"chmod\s+-R\s+777\s+/",                      # open permissions on root
        r">(\s*/dev/sd|\s*/dev/nvme)",                 # overwrite block device
        r"curl\s+.*\|\s*(ba)?sh",                      # pipe from internet to shell
        r"wget\s+.*\|\s*(ba)?sh",                      # pipe from internet to shell
        r"powershell.*-enc",                           # encoded powershell (obfuscation)
        r"format\s+[a-z]:",                            # Windows format drive
    ]
]

# Commands that get a stronger warning (still allowed after confirmation)
_WARN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"rm\s+-r", re.IGNORECASE), "Recursive delete"),
    (re.compile(r"git\s+push\s+.*--force", re.IGNORECASE), "Force push"),
    (re.compile(r"git\s+reset\s+--hard", re.IGNORECASE), "Hard reset"),
    (re.compile(r"DROP\s+(TABLE|DATABASE)", re.IGNORECASE), "SQL drop"),
    (re.compile(r"TRUNCATE\s+TABLE", re.IGNORECASE), "SQL truncate"),
    (re.compile(r"pip\s+install", re.IGNORECASE), "Package install"),
    (re.compile(r"npm\s+install", re.IGNORECASE), "Package install"),
    (re.compile(r"sudo\s+", re.IGNORECASE), "Elevated privileges"),
]

# Patterns that look like secrets in command output
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in [
        r"(?:api[_-]?key|apikey)\s*[:=]\s*[\w-]{20,}",
        r"(?:secret|token|password|passwd|pwd)\s*[:=]\s*\S{8,}",
        r"sk-[a-zA-Z0-9]{20,}",          # OpenAI key
        r"sk-ant-[a-zA-Z0-9-]{20,}",     # Anthropic key
        r"AIza[a-zA-Z0-9_-]{30,}",       # Google API key
        r"ghp_[a-zA-Z0-9]{30,}",         # GitHub PAT
        r"gho_[a-zA-Z0-9]{30,}",         # GitHub OAuth
        r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----",
        r"(?:AKIA|ASIA)[A-Z0-9]{16}",    # AWS access key
    ]
]


@dataclass
class SandboxResult:
    """Result of a sandbox safety check."""
    allowed: bool
    reason: str = ""
    warning: str = ""


class Sandbox:
    """Command sandboxing with blocklist and secret detection.

    Checks commands against a blocklist of dangerous patterns before
    execution, and scrubs potential secrets from command output before
    it enters the LLM conversation context.
    """

    def __init__(self, extra_blocked: list[str] | None = None) -> None:
        """Initialize sandbox.

        Args:
            extra_blocked: Additional regex patterns to block.
        """
        self._extra_blocked: list[re.Pattern[str]] = []
        if extra_blocked:
            self._extra_blocked = [
                re.compile(p, re.IGNORECASE) for p in extra_blocked
            ]

    def check_command(self, command: str) -> SandboxResult:
        """Check if a command is safe to execute.

        Returns:
            SandboxResult with allowed=False if blocked, warning if risky.
        """
        cmd_stripped = command.strip()

        # Check hard blocks
        for pattern in _BLOCKED_PATTERNS + self._extra_blocked:
            if pattern.search(cmd_stripped):
                logger.warning("Sandbox BLOCKED command: %s", cmd_stripped[:80])
                return SandboxResult(
                    allowed=False,
                    reason=f"Blocked by sandbox: matches dangerous pattern '{pattern.pattern}'",
                )

        # Check soft warnings
        warnings: list[str] = []
        for pattern, label in _WARN_PATTERNS:
            if pattern.search(cmd_stripped):
                warnings.append(label)

        if warnings:
            return SandboxResult(
                allowed=True,
                warning=f"[!] Risky operation: {', '.join(warnings)}",
            )

        return SandboxResult(allowed=True)

    def scrub_secrets(self, text: str) -> str:
        """Redact potential secrets from text.

        Replaces detected secret patterns with [REDACTED] to prevent
        API keys and credentials from leaking into the LLM context.

        Args:
            text: Raw text (typically command output).

        Returns:
            Text with secrets replaced by [REDACTED].
        """
        result = text
        for pattern in _SECRET_PATTERNS:
            result = pattern.sub("[REDACTED]", result)
        return result
