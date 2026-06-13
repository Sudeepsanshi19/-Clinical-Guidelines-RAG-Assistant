from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PID_PATH = ROOT / "storage" / "streamlit.pid"
LOG_PATH = ROOT / "storage" / "streamlit.log"
ERR_PATH = ROOT / "storage" / "streamlit.err.log"


def main() -> int:
    ROOT.joinpath("storage").mkdir(exist_ok=True)
    env = {
        "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
        "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
        "TEMP": os.environ.get("TEMP", r"C:\tmp"),
        "TMP": os.environ.get("TMP", r"C:\tmp"),
        "USERPROFILE": os.environ.get("USERPROFILE", r"C:\Users\sudee"),
        "APPDATA": os.environ.get("APPDATA", r"C:\Users\sudee\AppData\Roaming"),
        "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", r"C:\Users\sudee\AppData\Local"),
        "PYTHONUTF8": "1",
        "PYTHONPATH": str(ROOT / "src"),
        "PATH": os.pathsep.join(
            [
                str(Path(sys.executable).parent),
                str(Path(sys.executable).parent / "Scripts"),
                r"C:\Windows\System32",
                r"C:\Windows",
            ]
        ),
    }

    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with LOG_PATH.open("ab") as out, ERR_PATH.open("ab") as err:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/streamlit_app.py",
                "--server.port",
                "8502",
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
    PID_PATH.write_text(str(process.pid), encoding="utf-8")
    print(process.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

