from typing import Callable, Any


def start_end_log(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> None:
        print(f"{func.__name__} --- start.")
        func(*args, **kwargs)
        print(f"{func.__name__} --- done.")

    return wrapper
