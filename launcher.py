"""njss standalone desktop app entry point.

This module is the main entry point for the packaged (PyInstaller) version
of njss. It launches the GUI in native mode without requiring a terminal.
"""
import multiprocessing
import sys
import os
import traceback
from pathlib import Path

LOG = Path.home() / ".unjess" / "launch.log"


def _log(msg: str) -> None:
    """Append a timestamped line to the launch log."""
    import datetime
    with open(LOG, "a", encoding="utf-8") as f:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        f.write(f"[{ts}] {msg}\n")


def main() -> None:
    """Launch njss as a standalone desktop application."""
    multiprocessing.freeze_support()

    LOG.parent.mkdir(parents=True, exist_ok=True)

    # For child processes spawned by multiprocessing, don't mess with argv
    if "--multiprocessing-fork" in sys.argv:
        return

    # Clear old log
    LOG.write_text("", encoding="utf-8")

    # Redirect stderr to log file so we capture crashes
    if getattr(sys, 'frozen', False):
        os.environ['PYWEBVIEW_GUI'] = 'edgechromium'
        sys.stderr = open(LOG.parent / "stderr.log", "w", encoding="utf-8")

    sys.argv = [sys.argv[0], "--gui", "--native"]

    try:
        _log("Step 1: importing unjess.cli...")
        from unjess.cli import main as cli_main
        _log("Step 2: calling cli_main()")
        cli_main()
        _log("Step 3: cli_main() returned")
    except SystemExit as e:
        _log(f"SystemExit: code={e.code}")
    except Exception as exc:
        _log(f"EXCEPTION: {exc}")
        with open(LOG, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)


if __name__ == "__main__":
    main()
