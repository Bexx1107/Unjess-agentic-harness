"""Tests for unjess.mentions — @ mention parsing, resolution, and message processing."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unjess.mentions import Mention, MentionResolver


# ---------------------------------------------------------------------------
# Mention dataclass
# ---------------------------------------------------------------------------

class TestMention:
    """Tests for the Mention dataclass and its is_resolved property."""

    def test_is_resolved_false_when_empty(self) -> None:
        m = Mention(raw="@foo.py", kind="file", target="foo.py")
        assert m.is_resolved is False

    def test_is_resolved_false_when_blank_string(self) -> None:
        m = Mention(raw="@foo.py", kind="file", target="foo.py", resolved_content="")
        assert m.is_resolved is False

    def test_is_resolved_true_when_content_present(self) -> None:
        m = Mention(raw="@foo.py", kind="file", target="foo.py", resolved_content="hello")
        assert m.is_resolved is True

    def test_defaults(self) -> None:
        m = Mention(raw="@x", kind="file", target="x")
        assert m.resolved_content == ""

    def test_kind_stored(self) -> None:
        for kind in ("file", "dir", "git", "web", "url"):
            m = Mention(raw="@x", kind=kind, target="x")
            assert m.kind == kind


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParse:
    """Tests for MentionResolver.parse — regex-based mention extraction."""

    def test_parse_no_mentions_returns_empty(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        assert resolver.parse("Hello world, nothing special") == []

    def test_parse_file_mention(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("Look at @app.py please")
        assert len(mentions) == 1
        m = mentions[0]
        assert m.kind == "file"
        assert m.target == "app.py"
        assert m.raw == "@app.py"
        assert m.resolved_content == ""

    def test_parse_file_mention_simple_name(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("Check @README.md")
        assert len(mentions) == 1
        assert mentions[0].kind == "file"
        assert mentions[0].target == "README.md"

    def test_parse_dir_mention(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("List @src/ for me")
        assert len(mentions) == 1
        m = mentions[0]
        assert m.kind == "dir"
        assert m.target == "src/"
        assert m.raw == "@src/"

    def test_parse_dir_mention_nested(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("Show @path/to/dir/ contents")
        assert len(mentions) == 1
        assert mentions[0].kind == "dir"
        assert mentions[0].target == "path/to/dir/"

    def test_parse_git_diff(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("Show me @git diff")
        assert len(mentions) == 1
        m = mentions[0]
        assert m.kind == "git"
        assert m.target == "diff"
        assert m.raw == "@git diff"

    def test_parse_git_status(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("What is @git status?")
        assert len(mentions) == 1
        assert mentions[0].kind == "git"
        assert mentions[0].target == "status"

    def test_parse_git_log(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("Show @git log")
        assert len(mentions) == 1
        assert mentions[0].target == "log"

    def test_parse_git_branch(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("Which @git branch?")
        assert len(mentions) == 1
        assert mentions[0].target == "branch"

    def test_parse_web_mention(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse('Search @web "python dataclasses"')
        assert len(mentions) == 1
        m = mentions[0]
        assert m.kind == "web"
        assert m.target == "python dataclasses"
        assert m.raw == '@web "python dataclasses"'

    def test_parse_url_mention_http(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("Read @url http://example.com")
        assert len(mentions) == 1
        m = mentions[0]
        assert m.kind == "url"
        assert m.target == "http://example.com"

    def test_parse_url_mention_https(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mentions = resolver.parse("Fetch @url https://example.com/page?q=1")
        assert len(mentions) == 1
        m = mentions[0]
        assert m.kind == "url"
        assert m.target == "https://example.com/page?q=1"

    def test_parse_multiple_mentions(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        message = "Compare @app.py and @utils.py then check @src/"
        mentions = resolver.parse(message)
        kinds = {m.kind for m in mentions}
        assert "file" in kinds
        assert "dir" in kinds
        assert len(mentions) == 3

    def test_parse_mixed_kinds(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        message = 'Look at @README.md and @git diff and @web "test" and @url https://x.com'
        mentions = resolver.parse(message)
        kinds = sorted(m.kind for m in mentions)
        assert kinds == ["file", "git", "url", "web"]

    def test_parse_at_sign_alone_no_match(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        assert resolver.parse("email me @ home") == []

    def test_parse_git_unknown_subcommand_no_match(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        assert resolver.parse("@git rebase") == []

    def test_parse_web_without_quotes_no_match(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        assert resolver.parse("@web no quotes here") == []

    def test_parse_url_without_scheme_no_match(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        assert resolver.parse("@url www.example.com") == []


# ---------------------------------------------------------------------------
# Resolve — file
# ---------------------------------------------------------------------------

class TestResolveFile:
    """Tests for resolving file mentions."""

    def test_resolve_file_reads_content(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@README.md", kind="file", target="README.md")
        result = resolver.resolve([mention])
        assert len(result) == 1
        assert mention.is_resolved
        assert "# Test Project" in mention.resolved_content

    def test_resolve_file_nested_path(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@src/app.py", kind="file", target="src/app.py")
        resolver.resolve([mention])
        assert "def greet" in mention.resolved_content

    def test_resolve_file_not_found(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@missing.py", kind="file", target="missing.py")
        resolver.resolve([mention])
        assert "not found" in mention.resolved_content.lower()

    def test_resolve_file_is_directory(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@src", kind="file", target="src")
        resolver.resolve([mention])
        assert "not a file" in mention.resolved_content.lower()

    def test_resolve_file_too_large(self, tmp_workspace: Path) -> None:
        big_file = tmp_workspace / "big.txt"
        big_file.write_text("x" * 200_000, encoding="utf-8")
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@big.txt", kind="file", target="big.txt")
        resolver.resolve([mention])
        assert "too large" in mention.resolved_content.lower()

    def test_resolve_file_outside_workspace(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@../../etc/passwd", kind="file", target="../../etc/passwd")
        resolver.resolve([mention])
        assert "outside" in mention.resolved_content.lower()

    def test_resolve_file_wraps_in_code_block(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@README.md", kind="file", target="README.md")
        resolver.resolve([mention])
        assert "```" in mention.resolved_content
        assert "# README.md" in mention.resolved_content


# ---------------------------------------------------------------------------
# Resolve — directory
# ---------------------------------------------------------------------------

class TestResolveDir:
    """Tests for resolving directory mentions."""

    def test_resolve_dir_lists_children(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@src/", kind="dir", target="src/")
        resolver.resolve([mention])
        assert mention.is_resolved
        assert "app.py" in mention.resolved_content
        assert "utils.py" in mention.resolved_content

    def test_resolve_dir_skips_hidden_files(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "src" / ".hidden").write_text("secret", encoding="utf-8")
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@src/", kind="dir", target="src/")
        resolver.resolve([mention])
        assert ".hidden" not in mention.resolved_content

    def test_resolve_dir_not_found(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@nope/", kind="dir", target="nope/")
        resolver.resolve([mention])
        assert "not found" in mention.resolved_content.lower()

    def test_resolve_dir_is_actually_file(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@README.md/", kind="dir", target="README.md/")
        resolver.resolve([mention])
        # Should report it's not a directory
        assert mention.is_resolved  # error messages still count as resolved content

    def test_resolve_dir_shows_subdirectory_indicator(self, tmp_workspace: Path) -> None:
        sub = tmp_workspace / "parent" / "child"
        sub.mkdir(parents=True)
        (tmp_workspace / "parent" / "file.txt").write_text("hi", encoding="utf-8")
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@parent/", kind="dir", target="parent/")
        resolver.resolve([mention])
        assert "child/" in mention.resolved_content
        assert "file.txt" in mention.resolved_content

    def test_resolve_dir_shows_file_sizes(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@src/", kind="dir", target="src/")
        resolver.resolve([mention])
        assert "bytes" in mention.resolved_content

    def test_resolve_dir_outside_workspace(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@../../", kind="dir", target="../../")
        resolver.resolve([mention])
        assert "outside" in mention.resolved_content.lower()


# ---------------------------------------------------------------------------
# Resolve — git
# ---------------------------------------------------------------------------

class TestResolveGit:
    """Tests for resolving git mentions."""

    def test_resolve_git_no_git_integration(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace, git=None)
        mention = Mention(raw="@git diff", kind="git", target="diff")
        resolver.resolve([mention])
        assert "not a git repository" in mention.resolved_content.lower()

    def test_resolve_git_not_a_repo(self, tmp_workspace: Path) -> None:
        mock_git = MagicMock()
        mock_git.is_git_repo = False
        resolver = MentionResolver(tmp_workspace, git=mock_git)
        mention = Mention(raw="@git diff", kind="git", target="diff")
        resolver.resolve([mention])
        assert "not a git repository" in mention.resolved_content.lower()

    def test_resolve_git_diff_with_changes(self, tmp_workspace: Path) -> None:
        mock_git = MagicMock()
        mock_git.is_git_repo = True
        mock_git.diff.return_value = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"
        resolver = MentionResolver(tmp_workspace, git=mock_git)
        mention = Mention(raw="@git diff", kind="git", target="diff")
        resolver.resolve([mention])
        assert "+new" in mention.resolved_content
        mock_git.diff.assert_called_once()

    def test_resolve_git_diff_no_changes(self, tmp_workspace: Path) -> None:
        mock_git = MagicMock()
        mock_git.is_git_repo = True
        mock_git.diff.return_value = ""
        resolver = MentionResolver(tmp_workspace, git=mock_git)
        mention = Mention(raw="@git diff", kind="git", target="diff")
        resolver.resolve([mention])
        assert "no uncommitted changes" in mention.resolved_content.lower()

    def test_resolve_git_status(self, tmp_workspace: Path) -> None:
        mock_git = MagicMock()
        mock_git.is_git_repo = True
        mock_status = MagicMock()
        mock_status.summary.return_value = "on main — 2 modified"
        mock_git.status.return_value = mock_status
        resolver = MentionResolver(tmp_workspace, git=mock_git)
        mention = Mention(raw="@git status", kind="git", target="status")
        resolver.resolve([mention])
        assert "on main" in mention.resolved_content

    def test_resolve_git_log(self, tmp_workspace: Path) -> None:
        mock_git = MagicMock()
        mock_git.is_git_repo = True
        mock_commit = MagicMock()
        mock_commit.short_hash = "abc1234"
        mock_commit.message = "Initial commit"
        mock_commit.author = "Test"
        mock_commit.date = "2026-01-01"
        mock_git.log.return_value = [mock_commit]
        resolver = MentionResolver(tmp_workspace, git=mock_git)
        mention = Mention(raw="@git log", kind="git", target="log")
        resolver.resolve([mention])
        assert "abc1234" in mention.resolved_content
        assert "Initial commit" in mention.resolved_content
        mock_git.log.assert_called_once_with(n=10)

    def test_resolve_git_log_no_commits(self, tmp_workspace: Path) -> None:
        mock_git = MagicMock()
        mock_git.is_git_repo = True
        mock_git.log.return_value = []
        resolver = MentionResolver(tmp_workspace, git=mock_git)
        mention = Mention(raw="@git log", kind="git", target="log")
        resolver.resolve([mention])
        assert "no commits" in mention.resolved_content.lower()

    def test_resolve_git_branch(self, tmp_workspace: Path) -> None:
        mock_git = MagicMock()
        mock_git.is_git_repo = True
        mock_git.current_branch.return_value = "feature/cool"
        resolver = MentionResolver(tmp_workspace, git=mock_git)
        mention = Mention(raw="@git branch", kind="git", target="branch")
        resolver.resolve([mention])
        assert mention.resolved_content == "feature/cool"

    def test_resolve_git_branch_unknown(self, tmp_workspace: Path) -> None:
        mock_git = MagicMock()
        mock_git.is_git_repo = True
        mock_git.current_branch.return_value = ""
        resolver = MentionResolver(tmp_workspace, git=mock_git)
        mention = Mention(raw="@git branch", kind="git", target="branch")
        resolver.resolve([mention])
        assert "unknown branch" in mention.resolved_content.lower()

    def test_resolve_git_unknown_command(self, tmp_workspace: Path) -> None:
        mock_git = MagicMock()
        mock_git.is_git_repo = True
        resolver = MentionResolver(tmp_workspace, git=mock_git)
        mention = Mention(raw="@git rebase", kind="git", target="rebase")
        resolver.resolve([mention])
        assert "unknown git command" in mention.resolved_content.lower()


# ---------------------------------------------------------------------------
# Resolve — web / url
# ---------------------------------------------------------------------------

class TestResolveWebAndUrl:
    """Tests for resolving web search and URL fetch mentions."""

    @patch("unjess.mentions._search_web", create=True)
    def test_resolve_web_calls_search(self, _mock: MagicMock, tmp_workspace: Path) -> None:
        with patch("unjess.tools.web_tools._search_web", return_value="result1\nresult2"):
            resolver = MentionResolver(tmp_workspace)
            mention = Mention(raw='@web "python tips"', kind="web", target="python tips")
            resolver.resolve([mention])
            assert mention.is_resolved

    @patch("unjess.tools.web_tools._read_url", return_value="<page content>")
    def test_resolve_url_calls_read(self, mock_read: MagicMock, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@url https://example.com", kind="url", target="https://example.com")
        resolver.resolve([mention])
        assert mention.is_resolved
        mock_read.assert_called_once_with("https://example.com", max_length=10000)


# ---------------------------------------------------------------------------
# Resolve — error handling
# ---------------------------------------------------------------------------

class TestResolveErrorHandling:
    """Tests for exception handling in resolve."""

    def test_resolve_exception_produces_error_content(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        mention = Mention(raw="@foo.py", kind="file", target="foo.py")
        with patch.object(resolver, "_resolve_file", side_effect=RuntimeError("boom")):
            resolver.resolve([mention])
        assert "[Error resolving @foo.py: boom]" == mention.resolved_content

    def test_resolve_exception_does_not_stop_other_mentions(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        m1 = Mention(raw="@bad.py", kind="file", target="bad.py")
        m2 = Mention(raw="@README.md", kind="file", target="README.md")
        with patch.object(
            resolver,
            "_resolve_file",
            side_effect=[RuntimeError("fail"), "# README.md\n```\ncontent\n```"],
        ):
            result = resolver.resolve([m1, m2])
        assert len(result) == 2
        assert "Error" in m1.resolved_content
        assert m2.is_resolved


# ---------------------------------------------------------------------------
# process_message
# ---------------------------------------------------------------------------

class TestProcessMessage:
    """Tests for MentionResolver.process_message — end-to-end parse+resolve+inject."""

    def test_process_no_mentions_returns_original(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        msg, mentions = resolver.process_message("Just a normal message")
        assert msg == "Just a normal message"
        assert mentions == []

    def test_process_injects_file_content(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        msg, mentions = resolver.process_message("Read @README.md please")
        assert len(mentions) == 1
        assert "<mention" in msg
        assert "# Test Project" in msg
        assert "</mention>" in msg

    def test_process_preserves_original_message(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        msg, _ = resolver.process_message("Read @README.md please")
        assert msg.startswith("Read @README.md please")

    def test_process_multiple_mentions_injected(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        msg, mentions = resolver.process_message("Compare @src/app.py and @README.md")
        assert len(mentions) == 2
        assert msg.count("<mention") == 2
        assert msg.count("</mention>") == 2

    def test_process_unresolved_mention_not_injected(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        # A missing file still gets resolved_content (error message), so it counts
        # as resolved.  To get truly unresolved, patch _resolve_file to set empty.
        mention = Mention(raw="@ghost.py", kind="file", target="ghost.py")
        with patch.object(resolver, "parse", return_value=[mention]):
            with patch.object(resolver, "resolve", return_value=[mention]):
                msg, mentions = resolver.process_message("Look at @ghost.py")
        # mention.resolved_content is still "" so it's not injected
        assert "<mention" not in msg

    def test_process_returns_mentions_list(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        _, mentions = resolver.process_message("Check @src/app.py")
        assert all(isinstance(m, Mention) for m in mentions)


# ---------------------------------------------------------------------------
# Resolve returns same list
# ---------------------------------------------------------------------------

class TestResolveReturnValue:
    """Tests that resolve returns the same list it was given (mutated in-place)."""

    def test_resolve_returns_input_list(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        original = [Mention(raw="@README.md", kind="file", target="README.md")]
        result = resolver.resolve(original)
        assert result is original

    def test_resolve_empty_list(self, tmp_workspace: Path) -> None:
        resolver = MentionResolver(tmp_workspace)
        result = resolver.resolve([])
        assert result == []
