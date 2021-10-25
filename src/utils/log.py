from typing import Callable, Any
import logging
import logging.config

import yaml

LOGGING_CONFIG = "../configs/logging.yaml"


def load_config() -> None:
    with open(LOGGING_CONFIG, "r") as yml:
        logging.config.dictConfig(yaml.safe_load(yml))


def start_end_log(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> None:
        logging.debug(f"{func.__name__} --- start.")
        func(*args, **kwargs)
        logging.debug(f"{func.__name__} --- done.")

    return wrapper
