import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_BUDGET_S = 300.0  # competition limit: 5 minutes per prompt

def call_with_budget(fn: Callable[[], T], budget_s: float, fallback: Callable[[], T], label: str = "") -> tuple[T, bool]:
    """Run fn() with a wall-clock budget. If it does not finish in time, warn and
    return (fallback(), True); otherwise (result, False).

    Python cannot kill a thread, so a timed-out fn() keeps running in the background.
    Callers that share state with fn (e.g. a diffusers pipeline) should guard it with a
    lock so the next call waits for the abandoned one instead of running concurrently.
    """
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=budget_s), False
    except FuturesTimeout:
        warnings.warn(f"{label or 'generation'} exceeded {budget_s:.0f}s budget; using fallback renderer")
        return fallback(), True
    finally:
        ex.shutdown(wait=False)
