import sys
from typing import Any

from tqdm import tqdm


def progress_bar(*args: Any, **kwargs: Any) -> tqdm:
    """tqdm configured for reliable display in uvicorn/Windows terminals."""
    defaults: dict[str, Any] = {
        "file": sys.stdout,
        "dynamic_ncols": True,
        "mininterval": 0.5,
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    }
    defaults.update(kwargs)
    return tqdm(*args, **defaults)


def log_progress(message: str) -> None:
    """Print a status line without breaking active tqdm bars."""
    tqdm.write(message)
