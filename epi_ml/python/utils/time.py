from datetime import datetime

def time_now():
    """Return datetime of call without microseconds"""
    return datetime.utcnow().replace(microsecond=0)
