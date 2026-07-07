"""Tests for unjess.free_mode — free-tier model switching and usage tracking."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unjess.free_mode import (
    DailyUsage,
    FreeMode,
    FreeTier,
    UsageTracker,
    _FREE_TIERS,
    _load_latest_brain,
    _save_brain,
)


# ---------------------------------------------------------------------------
# FreeTier
# ---------------------------------------------------------------------------


class TestFreeTier:
    """Tests for FreeTier dataclass."""

    def test_has_limits_with_request_limit(self) -> None:
        tier = FreeTier(provider="p", model="m", display_name="M", daily_request_limit=100)
        assert tier.has_limits

    def test_has_limits_with_token_limit(self) -> None:
        tier = FreeTier(provider="p", model="m", display_name="M", daily_token_limit=1000)
        assert tier.has_limits

    def test_has_limits_with_both(self) -> None:
        tier = FreeTier(
            provider="p", model="m", display_name="M",
            daily_request_limit=10, daily_token_limit=1000,
        )
        assert tier.has_limits

    def test_no_limits(self) -> None:
        tier = FreeTier(provider="p", model="m", display_name="M")
        assert not tier.has_limits

    def test_default_fields(self) -> None:
        tier = FreeTier(provider="p", model="m", display_name="M")
        assert tier.daily_request_limit == 0
        assert tier.daily_token_limit == 0
        assert tier.rpm_limit == 0
        assert tier.priority == 0

    def test_free_tiers_list_not_empty(self) -> None:
        assert len(_FREE_TIERS) > 0

    def test_free_tiers_have_required_fields(self) -> None:
        for tier in _FREE_TIERS:
            assert tier.provider
            assert tier.model
            assert tier.display_name


# ---------------------------------------------------------------------------
# DailyUsage
# ---------------------------------------------------------------------------


class TestDailyUsage:
    """Tests for DailyUsage tracking."""

    def test_default_date_is_today(self) -> None:
        usage = DailyUsage(model="test-model")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert usage.date == today

    def test_explicit_date(self) -> None:
        usage = DailyUsage(model="m", date="2024-01-01")
        assert usage.date == "2024-01-01"

    def test_is_today_true(self) -> None:
        usage = DailyUsage(model="m")
        assert usage.is_today()

    def test_is_today_false(self) -> None:
        usage = DailyUsage(model="m", date="2020-01-01")
        assert not usage.is_today()

    def test_reset_if_new_day_resets_on_old_date(self) -> None:
        usage = DailyUsage(model="m", date="2020-01-01", requests=50, tokens=1000)
        usage.reset_if_new_day()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert usage.date == today
        assert usage.requests == 0
        assert usage.tokens == 0

    def test_reset_if_new_day_noop_on_today(self) -> None:
        usage = DailyUsage(model="m", requests=5, tokens=100)
        usage.reset_if_new_day()
        assert usage.requests == 5
        assert usage.tokens == 100

    def test_initial_counters(self) -> None:
        usage = DailyUsage(model="m")
        assert usage.requests == 0
        assert usage.tokens == 0


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------


class TestUsageTracker:
    """Tests for UsageTracker across all free-tier models."""

    def test_record_new_model(self) -> None:
        tracker = UsageTracker()
        tracker.record("model-a", tokens=100)
        usage = tracker.get_usage("model-a")
        assert usage.requests == 1
        assert usage.tokens == 100

    def test_record_increments(self) -> None:
        tracker = UsageTracker()
        tracker.record("model-a", tokens=50)
        tracker.record("model-a", tokens=75)
        usage = tracker.get_usage("model-a")
        assert usage.requests == 2
        assert usage.tokens == 125

    def test_get_usage_creates_entry(self) -> None:
        tracker = UsageTracker()
        usage = tracker.get_usage("new-model")
        assert usage.model == "new-model"
        assert usage.requests == 0
        assert usage.tokens == 0

    def test_remaining_pct_no_limits(self) -> None:
        tracker = UsageTracker()
        tier = FreeTier(provider="p", model="m", display_name="M")
        assert tracker.remaining_pct("m", tier) == 1.0

    def test_remaining_pct_request_limit(self) -> None:
        tracker = UsageTracker()
        tier = FreeTier(
            provider="p", model="m", display_name="M",
            daily_request_limit=100,
        )
        tracker.record("m")
        tracker.record("m")
        # 2 requests of 100 → 98% remaining
        pct = tracker.remaining_pct("m", tier)
        assert pct == pytest.approx(0.98)

    def test_remaining_pct_token_limit(self) -> None:
        tracker = UsageTracker()
        tier = FreeTier(
            provider="p", model="m", display_name="M",
            daily_token_limit=1000,
        )
        tracker.record("m", tokens=500)
        pct = tracker.remaining_pct("m", tier)
        assert pct == pytest.approx(0.5)

    def test_remaining_pct_both_limits_uses_min(self) -> None:
        tracker = UsageTracker()
        tier = FreeTier(
            provider="p", model="m", display_name="M",
            daily_request_limit=100,
            daily_token_limit=1000,
        )
        # 10 requests (90% remaining) and 800 tokens (20% remaining)
        for _ in range(10):
            tracker.record("m", tokens=80)
        pct = tracker.remaining_pct("m", tier)
        assert pct == pytest.approx(0.2)  # token limit is tighter

    def test_remaining_pct_over_limit(self) -> None:
        tracker = UsageTracker()
        tier = FreeTier(
            provider="p", model="m", display_name="M",
            daily_request_limit=2,
        )
        for _ in range(5):
            tracker.record("m")
        pct = tracker.remaining_pct("m", tier)
        assert pct < 0  # over limit

    def test_separate_models_tracked_independently(self) -> None:
        tracker = UsageTracker()
        tracker.record("model-a", tokens=100)
        tracker.record("model-b", tokens=200)
        assert tracker.get_usage("model-a").tokens == 100
        assert tracker.get_usage("model-b").tokens == 200


# ---------------------------------------------------------------------------
# Brain file
# ---------------------------------------------------------------------------


class TestBrainFile:
    """Tests for brain file save/load."""

    def test_save_brain_creates_file(self, tmp_path: Path) -> None:
        with patch("unjess.free_mode._BRAIN_DIR", tmp_path / "brain"):
            path = _save_brain("test summary", session_id="test123")
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "test summary" in content

    def test_save_brain_auto_session_id(self, tmp_path: Path) -> None:
        with patch("unjess.free_mode._BRAIN_DIR", tmp_path / "brain"):
            path = _save_brain("summary")
            assert path.suffix == ".md"
            assert path.exists()

    def test_load_latest_brain_empty_dir(self, tmp_path: Path) -> None:
        brain_dir = tmp_path / "brain"
        brain_dir.mkdir()
        with patch("unjess.free_mode._BRAIN_DIR", brain_dir):
            assert _load_latest_brain() == ""

    def test_load_latest_brain_no_dir(self, tmp_path: Path) -> None:
        with patch("unjess.free_mode._BRAIN_DIR", tmp_path / "nonexistent"):
            assert _load_latest_brain() == ""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        brain_dir = tmp_path / "brain"
        with patch("unjess.free_mode._BRAIN_DIR", brain_dir):
            _save_brain("hello world", session_id="001")
            content = _load_latest_brain()
            assert "hello world" in content

    def test_load_latest_picks_most_recent(self, tmp_path: Path) -> None:
        brain_dir = tmp_path / "brain"
        with patch("unjess.free_mode._BRAIN_DIR", brain_dir):
            _save_brain("first", session_id="aaa")
            _save_brain("second", session_id="zzz")
            content = _load_latest_brain()
            # 'zzz' sorts after 'aaa'
            assert "second" in content

    def test_load_latest_brain_truncates_long_content(self, tmp_path: Path) -> None:
        brain_dir = tmp_path / "brain"
        with patch("unjess.free_mode._BRAIN_DIR", brain_dir):
            long_text = "x" * 5000
            _save_brain(long_text, session_id="big")
            content = _load_latest_brain()
            assert len(content) <= 3000


# ---------------------------------------------------------------------------
# FreeMode
# ---------------------------------------------------------------------------


class TestFreeMode:
    """Tests for the FreeMode class."""

    def _make_free_mode(
        self,
        available_providers: set[str] | None = None,
    ) -> tuple[FreeMode, MagicMock, MagicMock, MagicMock]:
        """Create a FreeMode with mocked dependencies."""
        from unjess.config import Settings

        settings = Settings(model="gpt-4o", provider="openai")
        router = MagicMock()
        router.available_providers = available_providers or {"google", "groq"}
        display = MagicMock()
        display.show_info = MagicMock()
        display.show_error = MagicMock()
        display.show_warning = MagicMock()
        display.console = MagicMock()

        fm = FreeMode(settings, router, display)
        return fm, settings, router, display

    def test_initial_state(self) -> None:
        fm, _, _, _ = self._make_free_mode()
        assert not fm.is_active

    def test_activate_switches_model(self) -> None:
        fm, settings, _, display = self._make_free_mode({"google", "groq"})
        fm.activate()
        assert fm.is_active
        # Should have switched to a free-tier model
        assert settings.provider in {"google", "groq"}

    def test_activate_twice_shows_info(self) -> None:
        fm, _, _, display = self._make_free_mode({"google"})
        fm.activate()
        fm.activate()
        # Second call should show "already active"
        calls = [str(c) for c in display.show_info.call_args_list]
        assert any("already" in c.lower() for c in calls)

    def test_activate_no_cloud_keys(self) -> None:
        fm, _, _, display = self._make_free_mode({"ollama"})
        fm.activate()
        assert not fm.is_active
        display.show_error.assert_called_once()

    def test_deactivate_restores_original(self) -> None:
        fm, settings, _, _ = self._make_free_mode({"google"})
        original_model = settings.model
        original_provider = settings.provider
        fm.activate()
        fm.deactivate()
        assert not fm.is_active
        assert settings.model == original_model
        assert settings.provider == original_provider

    def test_deactivate_when_not_active(self) -> None:
        fm, _, _, display = self._make_free_mode()
        fm.deactivate()
        calls = [str(c) for c in display.show_info.call_args_list]
        assert any("not active" in c.lower() for c in calls)

    def test_check_and_switch_inactive(self) -> None:
        fm, _, _, _ = self._make_free_mode()
        result = fm.check_and_switch(tokens_used=100)
        assert result is None

    def test_check_and_switch_records_usage(self) -> None:
        fm, settings, _, _ = self._make_free_mode({"google", "groq"})
        fm.activate()
        fm.check_and_switch(tokens_used=500)
        current_model = settings.model
        usage = fm._tracker.get_usage(current_model)
        assert usage.tokens == 500
        assert usage.requests == 1

    @patch("unjess.free_mode._save_brain")
    def test_check_and_switch_switches_model_near_limit(
        self, mock_save: MagicMock,
    ) -> None:
        fm, settings, router, display = self._make_free_mode({"google", "groq"})
        fm.activate()
        model_before = settings.model

        # Find the active tier
        tier = fm._current_tier
        assert tier is not None

        # Exhaust the tier to below 10%
        if tier.daily_request_limit > 0:
            fm._tracker.get_usage(tier.model).requests = int(
                tier.daily_request_limit * 0.95
            )
        if tier.daily_token_limit > 0:
            fm._tracker.get_usage(tier.model).tokens = int(
                tier.daily_token_limit * 0.95
            )

        result = fm.check_and_switch(tokens_used=0)
        if tier.has_limits:
            # Should have tried to switch
            display.show_warning.assert_called()

    def test_check_and_switch_no_switch_when_quota_ok(self) -> None:
        fm, _, _, display = self._make_free_mode({"google", "groq"})
        fm.activate()
        result = fm.check_and_switch(tokens_used=1)
        assert result is None
        display.show_warning.assert_not_called()

    def test_rotate_key_no_pool(self) -> None:
        fm, settings, _, _ = self._make_free_mode()
        result = fm.rotate_key("google")
        assert result is False

    def test_rotate_key_single_key(self) -> None:
        fm, settings, _, _ = self._make_free_mode()
        settings.api_key_pool["google"] = ["key1"]
        result = fm.rotate_key("google")
        assert result is False

    def test_rotate_key_multiple_keys(self) -> None:
        fm, settings, _, display = self._make_free_mode()
        settings.api_key_pool["google"] = ["key1", "key2", "key3"]
        settings.api_keys["google"] = "key1"
        result = fm.rotate_key("google")
        assert result is True
        assert settings.api_keys["google"] == "key2"
        display.show_info.assert_called()

    def test_rotate_key_wraps_around(self) -> None:
        fm, settings, _, _ = self._make_free_mode()
        settings.api_key_pool["google"] = ["key1", "key2"]
        settings.api_keys["google"] = "key1"
        fm.rotate_key("google")
        assert settings.api_keys["google"] == "key2"
        fm.rotate_key("google")
        assert settings.api_keys["google"] == "key1"

    def test_build_switch_summary(self) -> None:
        fm, settings, _, _ = self._make_free_mode({"google"})
        fm.activate()
        summary = fm._build_switch_summary()
        assert "model switch" in summary.lower() or "Model switch" in summary
        assert settings.model in summary

    def test_activate_prefers_higher_priority(self) -> None:
        # Cerebras (priority=0) should be preferred over Google (priority=1)
        fm, settings, _, _ = self._make_free_mode({"cerebras", "google"})
        fm.activate()
        assert settings.provider == "cerebras"

    def test_activate_skips_ollama(self) -> None:
        fm, _, _, display = self._make_free_mode({"ollama"})
        fm.activate()
        # Should fail because _pick_best_cloud_tier excludes ollama
        assert not fm.is_active
        display.show_error.assert_called_once()


# ---------------------------------------------------------------------------
# FreeMode.show_status (smoke test)
# ---------------------------------------------------------------------------


class TestFreeModeShowStatus:
    """Smoke tests for status display."""

    def test_show_status_no_crash(self) -> None:
        from unjess.config import Settings

        settings = Settings(model="test", provider="google")
        router = MagicMock()
        router.available_providers = {"google"}
        display = MagicMock()
        display.console = MagicMock()

        fm = FreeMode(settings, router, display)
        # Should not raise
        fm.show_status()
        display.console.print.assert_called()
