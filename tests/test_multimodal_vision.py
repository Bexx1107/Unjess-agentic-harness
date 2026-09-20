"""Unit tests for multimodal image handling across providers."""

import base64
import json
from unittest.mock import MagicMock

from unjess.agent import Agent
from unjess.config import Settings
from unjess.llm.openai_compat import _prepare_messages_for_openai
from unjess.llm.anthropic_provider import _convert_messages_for_anthropic


def test_prepare_messages_for_openai_converts_images() -> None:
    sample_bytes = b"fake_png_data"
    b64_str = base64.b64encode(sample_bytes).decode("ascii")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this screenshot"},
                {"type": "image", "data": b64_str, "mime_type": "image/png"},
                {"type": "image", "data": sample_bytes, "mime_type": "image/jpeg"},  # raw bytes safety
            ],
        }
    ]

    prepared = _prepare_messages_for_openai(messages)
    # Must be JSON serializable
    serialized = json.dumps(prepared)
    assert "fake_png_data" not in serialized  # encoded
    assert f"data:image/png;base64,{b64_str}" in serialized
    assert f"data:image/jpeg;base64,{b64_str}" in serialized


def test_convert_messages_for_anthropic_converts_images() -> None:
    sample_bytes = b"fake_png_data"
    b64_str = base64.b64encode(sample_bytes).decode("ascii")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this screenshot"},
                {"type": "image", "data": b64_str, "mime_type": "image/png"},
            ],
        }
    ]

    converted = _convert_messages_for_anthropic(messages)
    serialized = json.dumps(converted)
    assert '"type": "image"' in serialized
    assert '"media_type": "image/png"' in serialized
    assert b64_str in serialized


def test_agent_run_stores_json_serializable_conversation() -> None:
    settings = Settings()
    router = MagicMock()
    router.chat_stream.return_value = []
    tools = MagicMock()
    tools.tool_names = []
    tools.get_schemas.return_value = []

    display = MagicMock()
    permissions = MagicMock()
    agent = Agent(settings=settings, router=router, tool_registry=tools, display=display, permissions=permissions)
    test_img = {
        "data": base64.b64encode(b"hello_image").decode("ascii"),
        "mime_type": "image/png",
        "name": "screenshot.png",
    }

    agent.run("look at this", images=[test_img])

    # self._conversation must be 100% JSON-serializable without raising TypeError
    dumped = json.dumps(agent.conversation)
    assert "hello_image" not in dumped  # base64 string
    assert "image" in dumped
