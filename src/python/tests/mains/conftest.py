"""Pytest configuration"""
# pylint: disable=unused-argument
import os
import subprocess
from pathlib import Path

os.environ["COMET_AUTO_LOG_DISABLE"] = "True"
os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
os.environ["COMET_FALLBACK_STREAMER_KEEP_OFFLINE_ZIP"] = "False"


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    current_dir = Path(__file__).parent.resolve()

    to_rm = current_dir / "*.pickle"
    subprocess.run(args=f"rm -f {to_rm}", shell=True, check=True)
