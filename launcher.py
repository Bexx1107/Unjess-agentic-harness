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
        # Add host Python site-packages so the desktop app can import external libraries (playwright, etc.)
        import glob
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        app_data = os.environ.get("APPDATA", "")
        if local_app_data:
            os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", os.path.join(local_app_data, "ms-playwright"))
        candidates = []
        if local_app_data:
            candidates.extend(glob.glob(os.path.join(local_app_data, "Programs", "Python", "Python*", "Lib", "site-packages")))
        if app_data:
            candidates.extend(glob.glob(os.path.join(app_data, "Python", "Python*", "site-packages")))
        for p in os.environ.get("PATH", "").split(os.pathsep):
            if "python" in p.lower():
                for sub in (os.path.join("Lib", "site-packages"), "site-packages"):
                    sp = os.path.join(p, sub)
                    if os.path.isdir(sp):
                        candidates.append(sp)
                    parent_sp = os.path.join(os.path.dirname(p), sub)
                    if os.path.isdir(parent_sp):
                        candidates.append(parent_sp)
        for c in candidates:
            if c not in sys.path and os.path.isdir(c):
                sys.path.append(c)

    sys.argv = [sys.argv[0], "--gui", "--native"]

    try:
        _log("Step 1: importing unjess.cli...")
        import asyncio
        import nest_asyncio
        nest_asyncio.apply()
        _orig_asyncio_run = asyncio.run
        def _safe_asyncio_run(main, *args, **kwargs):
            kwargs.pop("loop_factory", None)
            return _orig_asyncio_run(main, *args, **kwargs)
        asyncio.run = _safe_asyncio_run

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
