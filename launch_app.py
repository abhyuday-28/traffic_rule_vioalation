from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tkinter import Tk, messagebox


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def show_error(message: str) -> None:
    root = Tk()
    root.withdraw()
    messagebox.showerror("Traffic Violation System", message)
    root.destroy()


def main() -> int:
    root_dir = project_root()
    python_exe = root_dir / "venv" / "Scripts" / "python.exe"
    pythonw_exe = root_dir / "venv" / "Scripts" / "pythonw.exe"
    main_py = root_dir / "main.py"
    log_path = root_dir / "launcher_error.log"

    if not python_exe.exists():
        show_error(f"Missing venv Python:\n{python_exe}")
        return 1
    if not main_py.exists():
        show_error(f"Missing main.py:\n{main_py}")
        return 1

    interpreter = pythonw_exe if pythonw_exe.exists() else python_exe
    try:
        result = subprocess.run(
            [str(interpreter), str(main_py)],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        show_error(str(exc))
        return 1
    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "Unknown launcher error."
        log_path.write_text(error_text, encoding="utf-8")
        show_error(f"Application failed to start.\n\nDetails saved to:\n{log_path}")
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
