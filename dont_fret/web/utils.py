from typing import Optional, TypeVar
import time

T = TypeVar("T")


def not_none(*args: Optional[T]) -> T:
    """Returns the first `not None` argument"""
    for a in args:
        if a is not None:
            return a
    raise ValueError("All arguments are `None`")


def some_task(param=3) -> int:
    time.sleep(param / 2)
    return param * 2
