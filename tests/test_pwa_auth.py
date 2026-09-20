"""Unit tests for PWA (Progressive Web App) and PIN Authentication."""

import pytest
from unjess.config import Settings
from unjess.gui.pwa import _MANIFEST_CONTENT, _SERVICE_WORKER_JS, _PWA_HEAD_HTML
from unjess.gui.auth import is_authenticated


def test_pwa_manifest_structure() -> None:
    assert _MANIFEST_CONTENT["name"] == "Unjess AI"
    assert _MANIFEST_CONTENT["display"] == "standalone"
    assert _MANIFEST_CONTENT["start_url"] == "/"
    assert len(_MANIFEST_CONTENT["icons"]) > 0


def test_pwa_service_worker_content() -> None:
    assert "self.skipWaiting()" in _SERVICE_WORKER_JS
    assert "addEventListener('install'" in _SERVICE_WORKER_JS


def test_pwa_head_html() -> None:
    assert 'rel="manifest"' in _PWA_HEAD_HTML
    assert 'apple-mobile-web-app-capable' in _PWA_HEAD_HTML


def test_is_authenticated_when_pin_empty() -> None:
    settings = Settings(mobile_pin="")
    # When no PIN set, everyone is authenticated by default
    assert is_authenticated(settings) is True


def test_settings_mobile_companion_fields() -> None:
    settings = Settings(mobile_pin="1234", allow_network_access=True, network_port=8080)
    assert settings.mobile_pin == "1234"
    assert settings.allow_network_access is True
    assert settings.network_port == 8080
