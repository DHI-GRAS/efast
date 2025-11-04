import logging
import time
from collections import namedtuple
from typing import Callable, Tuple

TimingsResult = namedtuple("TimingsResult", ['time', 'result'])

def timed_call(f: Callable, *args, **kwargs):

    start = time.perf_counter()
    res = f(*args, **kwargs)
    end = time.perf_counter()

    return TimingsResult(time=end - start, result = res)

class Timer:
    def __init__(self, logger=None, dsc: str="", add_to: list[Tuple[str, float]]|None=None):
        if logger is True:
            logger = logging.getLogger("Timer")
        self.logger = logger
        self.dsc = dsc
        self.add_to = add_to

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = time.perf_counter() - self.start
        if self.logger is not None:
            self.logger.info(f"[Timed call] {self.elapsed:.4f}s {self.dsc}")
        if self.add_to is not None:
            self.add_to.append((self.dsc, self.elapsed))
