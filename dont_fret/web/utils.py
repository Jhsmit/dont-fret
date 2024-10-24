import time
from typing import Optional, TypeVar

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


def find_object(items: list[T], **kwargs) -> T:
    for item in items:
        if all(getattr(item, key) == value for key, value in kwargs.items()):
            return item
    raise ValueError("Object not found")


def find_index(items: list, **kwargs) -> int:
    for i, item in enumerate(items):
        if all(getattr(item, key) == value for key, value in kwargs.items()):
            return i
    raise ValueError("Object not found")
