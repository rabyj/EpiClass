"""Pytest configuration"""
# pylint: disable=unused-argument
import subprocess
from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    current_dir = Path(__file__).parent.resolve()

    to_rm = current_dir / "*.pickle"
    subprocess.run(args=f"rm -f {to_rm}", shell=True, check=True)
