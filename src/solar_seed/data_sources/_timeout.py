"""
Timeout Helper for Blocking Network Calls
=========================================

Run a blocking call (e.g. Fido.search / Fido.fetch) in a worker thread with
a hard timeout that actually returns control to the caller.

Why not ``with ThreadPoolExecutor(...) as executor``?
The context manager calls ``shutdown(wait=True)`` on exit, which blocks
until the worker thread finishes — so even after ``future.result(timeout=...)``
raises TimeoutError, the caller would still hang until the stuck network
call completes. That makes the timeout useless.

Tradeoff: on timeout the worker thread is deliberately orphaned (Python
cannot kill threads). It keeps running in the background until the
underlying network call finishes or errors out on its own; until then it
holds its socket/memory and may delay interpreter exit at process shutdown.
This is accepted — an orphaned thread is better than a hung monitoring loop.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

__all__ = ["run_with_timeout", "FutureTimeoutError"]


def run_with_timeout(fn, timeout: float, label: str = ""):
    """
    Run ``fn()`` in a worker thread, waiting at most ``timeout`` seconds.

    Args:
        fn: Zero-argument callable (wrap args in a lambda/closure).
        timeout: Maximum seconds to wait for the result.
        label: Optional label for the worker thread name (diagnostics).

    Returns:
        Whatever ``fn()`` returns.

    Raises:
        concurrent.futures.TimeoutError: If ``fn`` does not finish in time.
            The worker thread is abandoned (see module docstring for tradeoff).
        Exception: Any exception raised by ``fn`` is re-raised.
    """
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=f"timeout-{label}" if label else "timeout",
    )
    future = executor.submit(fn)
    try:
        result = future.result(timeout=timeout)
    except FutureTimeoutError:
        # Do NOT wait for the hung worker — abandon it (see module docstring).
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except BaseException:
        # fn() raised: worker is done, non-blocking shutdown is safe.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return result
