"""Utilities for consistently using the repository-local PEFT checkout."""

from pathlib import Path
import sys


def ensure_local_peft_first() -> Path | None:
    """Put ``<repo>/peft/src`` at the front of ``sys.path`` when it exists."""
    repo_root = Path(__file__).resolve().parent.parent
    local_peft_src = repo_root / "peft" / "src"
    if not local_peft_src.exists():
        return None

    local_peft_src_str = str(local_peft_src)
    if sys.path[:1] != [local_peft_src_str]:
        try:
            sys.path.remove(local_peft_src_str)
        except ValueError:
            pass
        sys.path.insert(0, local_peft_src_str)
    return local_peft_src
