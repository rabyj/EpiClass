"""Check for specific conditions before running the main program, .e.g git in clean state, metadata passes all test flags, etc.

Uses GitPython package.
"""
from __future__ import annotations

import sys
from pathlib import Path

from git import Repo


def find_upwards(cwd: Path, filename: str) -> Path | None:
    """Python: search for a file in current directory and all it's parents
    https://stackoverflow.com/a/70859914/11472153"""
    if cwd == Path(cwd.root) or cwd == cwd.parent:
        return None

    fullpath = cwd / filename

    return fullpath if fullpath.exists() else find_upwards(cwd.parent, filename)


def git_repo_is_clean(verbose=True) -> bool:
    """Verifies that the git has no staged or tracked unstaged changes.

    Return True if condition verified.
    """
    is_clean = True
    dir_here = Path(__file__).resolve()
    git_dir = find_upwards(dir_here, ".git")
    if git_dir is None:
        raise OSError("Could not find git directory.")

    repo = Repo(git_dir)

    # Verify if staged changes exist
    staged = [item.a_path for item in repo.index.diff("HEAD")]
    if staged:
        is_clean = False
        if verbose:
            print(f"Staged files exist: {staged}")

    # Verify if tracked+unstaged changes exist
    unstaged = [item.a_path for item in repo.index.diff(None)]
    if unstaged:
        is_clean = False
        if verbose:
            print(f"Tracked+unstaged changes exist: {unstaged}")

    return is_clean


def main():
    """Exit with error a precondition is not met."""

    if not git_repo_is_clean():
        print("Git directory not in a clean state, exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
