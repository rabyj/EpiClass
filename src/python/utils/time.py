from datetime import datetime


def time_now() -> datetime:
    """Return datetime of call without microseconds"""
    return datetime.utcnow().replace(microsecond=0)


def time_now_str() -> str:
    return time_now().strftime("%Y-%m-%d_%H-%M-%S")
