"""Time utilities"""

from datetime import datetime


def time_now() -> datetime:
    """Return datetime of call without microseconds"""
    return datetime.utcnow().replace(microsecond=0)


def time_now_str() -> str:
    """Return datetime of call as a string in the format YYYY-MM-DD_HH-MM-SS"""
    return time_now().strftime("%Y-%m-%d_%H-%M-%S")


def seconds_to_str(seconds: int) -> str:
    """Convert seconds to a string in the format HH:MM:SS"""
    return str(datetime.utcfromtimestamp(seconds).strftime("%H:%M:%S"))
