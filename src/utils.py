from pathlib import Path
import subprocess


def get_repo_root() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
