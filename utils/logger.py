import logging
from datetime import datetime


def get_time(fmt='%Y%m%d_%H%M%S') -> str:
    time_stamp = datetime.now().strftime(fmt)
    return time_stamp


def save_config(args) -> None:
    ...
