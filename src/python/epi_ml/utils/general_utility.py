"""For lost little functions that don't fit anywhere else."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List


def write_hdf5_paths_to_file(
    md5s: Iterable[str], parent: str, suffix: str, filepath: str | Path
) -> List[Path]:
    """Write a list of md5s to a file, with a prefix and suffix.
    Creates files of the format: Path(prefix=parent)/{md5}_{suffix}.hdf5

    Returns a list of the paths written to the file.
    """
    files = []
    with open(filepath, "w", encoding="utf8") as f:
        for md5 in md5s:
            path = Path(parent) / (f"{md5}_{suffix}.hdf5")
            f.write(f"{path}\n")
            files.append(path)
    return files


def write_md5s_to_file(md5s: Iterable[str], logdir: str | Path, name: str) -> Path:
    """Write a list of md5s to a file. Return filepath."""
    filename = Path(logdir) / f"{name}.md5"
    with open(filename, "w", encoding="utf8") as f:
        for md5 in md5s:
            f.write(f"{md5}\n")
    return filename


def get_valid_filename(name: str):
    """Tranform a string into a valid filename."""
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "_", s)
    s = re.sub(r"(_{2,})", "_", s)
    s = s.replace("_-_", "-")
    if s[-1] == "_":
        s = s[:-1]
    return s
