"""OAuth provider — browser-based OAuth login flow for subscription services.

Supports ChatGPT Plus and similar services that require browser-based
authentication and session token capture.
"""

import json
import logging
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

from unjess.llm.base import LLMProvider, LLMResponse, StreamChunk, ToolCall, Usage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@dataclass
class OAuthSession:
    """A stored OAuth session with tokens."""

    provider: str
    access_token: str = ""
    refresh_token: str = ""
    session_token: str = ""
    expires_at: float = 0.0
    user_agent: str = ""
    cookies: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Whether the session has expired."""
        if not self.expires_at:
            return False
        return time.time() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Whether the session has a usable token."""
        return bool(self.access_token or self.session_token) and not self.is_expired


class SessionManager:
    """Manages OAuth session storage, loading, and refresh.

    Sessions are stored as JSON files in the config directory.

    Args:
        storage_dir: Directory for session files.
    """

    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir
        self._sessions: dict[str, OAuthSession] = {}

    def get(self, provider: str) -> Optional[OAuthSession]:
        """Get a stored session for a provider.

        Args:
            provider: Provider name (e.g., "chatgpt-plus").

        Returns:
            OAuthSession if found and valid, None otherwise.
        """
        # Check memory cache
        session = self._sessions.get(provider)
        if session and session.is_valid:
            return session

        # Try loading from disk
        session = self._load(provider)
        if session and session.is_valid:
            self._sessions[provider] = session
            return session

        return None

    def store(self, session: OAuthSession) -> None:
        """Store a session to disk.

        Args:
            session: The session to store.
        """
        self._sessions[session.provider] = session
        self._save(session)

    def delete(self, provider: str) -> bool:
        """Delete a stored session.

        Args:
            provider: Provider name.

        Returns:
            True if deleted.
        """
        self._sessions.pop(provider, None)
        path = self._session_path(provider)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_providers(self) -> list[str]:
        """List all providers with stored sessions."""
        providers: list[str] = []
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        for path in self._storage_dir.glob("*.json"):
            providers.append(path.stem)
        return providers

    # ----- Internal -----

    def _session_path(self, provider: str) -> Path:
        """Get the file path for a provider's session."""
        return self._storage_dir / f"{provider}.json"

    def _save(self, session: OAuthSession) -> None:
        """Save a session to disk."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._session_path(session.provider)

        data = {
            "provider": session.provider,
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "session_token": session.session_token,
            "expires_at": session.expires_at,
            "user_agent": session.user_agent,
            "cookies": session.cookies,
            "metadata": session.metadata,
        }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Restrict file permissions — tokens are sensitive
        try:
            if os.name != 'nt':
                # Unix: owner read/write only
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
            else:
                # Windows: restrict to current user via icacls (best effort)
                import subprocess
                username = os.environ.get('USERNAME', '')
                if username:
                    subprocess.run(
                        ['icacls', str(path), '/inheritance:r', '/grant:r', f'{username}:(R,W)'],
                        capture_output=True, timeout=5,
                    )
        except Exception:
            logger.debug("Could not restrict permissions on %s", path)

        logger.debug("Session saved for %s", session.provider)

    def _load(self, provider: str) -> Optional[OAuthSession]:
        """Load a session from disk."""
        path = self._session_path(provider)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return OAuthSession(**data)
        except Exception as exc:
            logger.warning("Failed to load session for %s: %s", provider, exc)
            return None


# ---------------------------------------------------------------------------
# OAuth LLM Provider
# ---------------------------------------------------------------------------

class OAuthProvider(LLMProvider):
    """LLM provider that uses OAuth/session tokens instead of API keys.

    Authenticates via browser-based OAuth flow and uses internal
    API endpoints (e.g., ChatGPT's backend API).

    Args:
        provider_name: Human-readable name.
        auth_url: URL to open for OAuth login.
        api_base: Internal API base URL.
        session_manager: SessionManager for token storage.
        models: List of available models.
    """

    def __init__(
        self,
        provider_name: str = "chatgpt-plus",
        auth_url: str = "https://chat.openai.com",
        api_base: str = "https://chat.openai.com/backend-api",
        session_manager: Optional[SessionManager] = None,
        models: list[str] | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._auth_url = auth_url
        self._api_base = api_base
        self._session_manager = session_manager
        self._models = models or ["gpt-4o"]

    @property
    def provider_name(self) -> str:
        """Provider name."""
        return self._provider_name

    @property
    def available_models(self) -> list[str]:
        """Available models."""
        return self._models

    def is_configured(self) -> bool:
        """Whether the provider has a valid session."""
        if self._session_manager:
            session = self._session_manager.get(self._provider_name)
            return session is not None and session.is_valid
        return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> LLMResponse:
        """Send a chat request using the OAuth session.

        Note: This is a stub — full implementation requires
        reverse-engineering the internal API for each provider.

        Args:
            messages: Conversation messages.
            tools: Tool definitions.
            model: Model name.

        Returns:
            LLMResponse.
        """
        session = None
        if self._session_manager:
            session = self._session_manager.get(self._provider_name)

        if not session or not session.is_valid:
            return LLMResponse(
                text="OAuth session expired or not found. Run /auth to re-authenticate.",
                usage=Usage(),
            )

        # Stub: actual implementation would use httpx/aiohttp
        # to call the internal API with the session token
        logger.warning(
            "OAuth provider '%s' chat() is not fully implemented. "
            "Use the direct API provider instead.",
            self._provider_name,
        )

        return LLMResponse(
            text="[OAuth provider not fully implemented — use direct API]",
            usage=Usage(),
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
    ) -> Generator[StreamChunk, None, None]:
        """Stream a response (stub — yields a single done chunk)."""
        response = self.chat(messages, tools, model)
        yield StreamChunk(text=response.text)
        yield StreamChunk(done=True, usage=response.usage)

    def authenticate(self) -> bool:
        """Launch browser-based OAuth authentication.

        Opens the auth URL in the user's browser and waits for
        the session token to be captured.

        Returns:
            True if authentication succeeded.
        """
        token = openai_oauth_login()
        if token and self._session_manager:
            session = OAuthSession(
                provider=self._provider_name,
                access_token=token,
                expires_at=time.time() + 3600,  # 1 hour default
            )
            self._session_manager.store(session)
            return True
        return False


# ---------------------------------------------------------------------------
# OpenAI OAuth PKCE Login (Experimental)
# ---------------------------------------------------------------------------

# Same client ID and endpoints used by the Codex CLI
_OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
_OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
_OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_OPENAI_REDIRECT_PORT = 1455
_OPENAI_REDIRECT_URI = f"http://localhost:{_OPENAI_REDIRECT_PORT}/auth/callback"
_OPENAI_SCOPES = "openid profile email offline_access"


def _generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code verifier and challenge (S256).

    Returns:
        Tuple of (code_verifier, code_challenge).
    """
    import base64
    import hashlib
    import secrets

    # RFC 7636: 43-128 characters from [A-Z, a-z, 0-9, -, ., _, ~]
    verifier = secrets.token_urlsafe(64)[:128]
    # S256: BASE64URL(SHA256(verifier))
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def openai_oauth_login(timeout: int = 120) -> Optional[str]:
    """Perform OpenAI OAuth PKCE login via browser.

    Experimental — uses the same OAuth flow as the Codex CLI.

    1. Generates PKCE code verifier/challenge
    2. Starts a local HTTP server on port 1455
    3. Opens the browser to auth.openai.com
    4. Waits for the OAuth callback with the authorization code
    5. Exchanges the code for an access token

    Args:
        timeout: Seconds to wait for the callback. Default: 120.

    Returns:
        Access token string on success, None on failure.
    """
    import http.server
    import secrets
    import urllib.parse
    import webbrowser

    verifier, challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(32)

    # Build authorization URL
    auth_params = urllib.parse.urlencode({
        "client_id": _OPENAI_CLIENT_ID,
        "redirect_uri": _OPENAI_REDIRECT_URI,
        "response_type": "code",
        "scope": _OPENAI_SCOPES,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })
    auth_url = f"{_OPENAI_AUTH_URL}?{auth_params}"

    # Result container
    result: dict[str, Optional[str]] = {"code": None, "error": None}

    # --- Local callback server ---
    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        """Handles the OAuth redirect callback."""

        def do_GET(self) -> None:
            """Process the OAuth callback."""
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if parsed.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                return

            # Verify state
            received_state = params.get("state", [""])[0]
            if received_state != state:
                result["error"] = "State mismatch — possible CSRF attack"
                self._respond("Login failed: state mismatch. Close this tab.")
                return

            # Check for error
            if "error" in params:
                result["error"] = params.get("error_description", params["error"])[0]
                self._respond(f"Login failed: {result['error']}. Close this tab.")
                return

            # Capture authorization code
            code = params.get("code", [""])[0]
            if code:
                result["code"] = code
                self._respond(
                    "Login successful! You can close this tab and return to njss."
                )
            else:
                result["error"] = "No authorization code received"
                self._respond("Login failed: no code. Close this tab.")

        def _respond(self, message: str) -> None:
            """Send a simple HTML response."""
            html = (
                "<!DOCTYPE html><html><body style='font-family:sans-serif;"
                "display:flex;justify-content:center;align-items:center;"
                "height:100vh;margin:0;background:#1a1a2e;color:#e0e0e0'>"
                f"<h2>{message}</h2></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, format: str, *args: object) -> None:
            """Suppress HTTP request logging."""
            pass

    # Start server
    try:
        server = http.server.HTTPServer(
            ("127.0.0.1", _OPENAI_REDIRECT_PORT), CallbackHandler
        )
    except OSError as exc:
        logger.error("Failed to start callback server on port %d: %s",
                      _OPENAI_REDIRECT_PORT, exc)
        return None

    server.timeout = timeout

    # Open browser
    logger.info("Opening browser for OAuth login: %s", auth_url[:80])
    print(f"  Auth URL: {auth_url[:120]}...")
    webbrowser.open(auth_url)

    # Wait for callback (single request)
    server.handle_request()
    server.server_close()

    if result["error"]:
        logger.error("OAuth login failed: %s", result["error"])
        return None

    if not result["code"]:
        logger.error("OAuth login timed out — no callback received")
        return None

    # --- Exchange authorization code for access token ---
    try:
        import urllib.request

        token_data = urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "client_id": _OPENAI_CLIENT_ID,
            "code": result["code"],
            "redirect_uri": _OPENAI_REDIRECT_URI,
            "code_verifier": verifier,
        }).encode()

        req = urllib.request.Request(
            _OPENAI_TOKEN_URL,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            token_response = json.loads(resp.read().decode())

        access_token = token_response.get("access_token", "")
        if not access_token:
            logger.error("Token response missing access_token: %s",
                          list(token_response.keys()))
            return None

        # Store token info
        logger.info("OAuth login successful — token expires in %ss",
                      token_response.get("expires_in", "unknown"))
        return access_token

    except Exception as exc:
        logger.error("Token exchange failed: %s", exc)
        return None

