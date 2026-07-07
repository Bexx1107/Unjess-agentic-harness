"""njss Desktop App Launcher.

Double-click this file (or create a shortcut to it) to launch njss
as a native desktop application. No terminal window will appear.

On Windows, .pyw files are run with pythonw.exe by default, which
suppresses the console window — making it feel like a real app.
"""
from unjess.cli import gui_main

if __name__ == "__main__":
    gui_main()
