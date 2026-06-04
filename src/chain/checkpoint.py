"""
src/chain/checkpoint.py — Atomic checkpoint save/load for the data pipeline.

Checkpoints allow a long multi-hour run to resume from where it left off
without restarting Stage 1 if Stage 3 crashes. Each stage saves its output
as a JSON checkpoint before writing the CSV output.

Atomicity guarantee: write to a temp file first, then os.rename() — which is
atomic on POSIX systems. A crash mid-write leaves a .tmp file, not a corrupt
checkpoint. On next run, the .tmp file is ignored (not loaded).
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path(config.CHECKPOINT_DIR)


def _ensure_dir() -> None:
    """Create checkpoint directory if it does not exist."""
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _path(stage_name: str) -> Path:
    """Return the canonical path for a stage checkpoint file."""
    return _CHECKPOINT_DIR / f"{stage_name}.json"


def save_checkpoint(stage_name: str, data: Any) -> None:
    """
    Atomically serialise `data` to a JSON checkpoint file.

    The write is atomic: data is written to a .tmp file in the same directory,
    then renamed to the final path. A crash mid-write never corrupts the
    previous valid checkpoint.

    Args:
        stage_name: Unique name for this pipeline stage
                    (e.g. "stage1_graduations").
        data:       JSON-serialisable Python object (dict, list, etc.).

    Raises:
        TypeError: If `data` is not JSON-serialisable.
        OSError:   On filesystem errors.
    """
    _ensure_dir()
    final_path = _path(stage_name)

    # Write to a temp file in the SAME directory so os.rename() is atomic
    # (rename across filesystems is not atomic).
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=_CHECKPOINT_DIR,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp_file:
            tmp_path = tmp_file.name
            json.dump(data, tmp_file, indent=2, default=str)

        # Atomic rename: replaces final_path if it already exists.
        os.rename(tmp_path, final_path)
        logger.info(
            "Checkpoint saved: %s (%d items)",
            final_path,
            len(data) if hasattr(data, "__len__") else 1,
        )
    except Exception:
        # Clean up temp file if something went wrong before the rename.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(stage_name: str) -> Optional[Any]:
    """
    Load and deserialise a checkpoint if it exists.

    Args:
        stage_name: Stage name matching what was passed to save_checkpoint().

    Returns:
        The deserialised Python object if the checkpoint exists, else None.

    Raises:
        json.JSONDecodeError: If the checkpoint file is corrupt (written before
                              atomicity was added, or disk corruption).
    """
    path = _path(stage_name)
    if not path.exists():
        logger.debug("No checkpoint found for stage %r.", stage_name)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "Checkpoint loaded: %s (%d items)",
            path,
            len(data) if hasattr(data, "__len__") else 1,
        )
        return data
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"Checkpoint file {path} is corrupt. "
            "Delete it and re-run the stage to regenerate it.\n"
            f"Original error: {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc


def checkpoint_exists(stage_name: str) -> bool:
    """Return True if a valid checkpoint file exists for this stage."""
    return _path(stage_name).exists()


def clear_checkpoint(stage_name: str) -> None:
    """
    Delete the checkpoint for a stage, forcing it to re-run on the next
    pipeline execution.

    Args:
        stage_name: Stage name to clear.

    Raises:
        FileNotFoundError: If the checkpoint does not exist (i.e. nothing to clear).
    """
    path = _path(stage_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint to clear for stage {stage_name!r} (path: {path})."
        )
    path.unlink()
    logger.info("Checkpoint cleared: %s", path)


def list_checkpoints() -> list[str]:
    """Return the names of all existing checkpoint stages."""
    _ensure_dir()
    return [
        p.stem
        for p in sorted(_CHECKPOINT_DIR.glob("*.json"))
    ]
