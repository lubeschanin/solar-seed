"""
Timeout Helper for Blocking Network Calls
=========================================

Run a blocking call (e.g. Fido.search / Fido.fetch) in a worker thread with
a hard timeout that actually returns control to the caller.

Why a raw daemon thread and not ``ThreadPoolExecutor``?
Two of the executor's guarantees work against us here:

1. ``with ThreadPoolExecutor(...)`` calls ``shutdown(wait=True)`` on exit,
   which blocks until the worker finishes — so even after
   ``future.result(timeout=...)`` raises TimeoutError the caller would hang
   until the stuck network call completes. That makes the timeout useless.
2. Executor workers are NON-daemon threads and are additionally joined by an
   interpreter-shutdown hook. Abandoning one therefore moves the hang from
   the call site to process exit: the backfill cron aborted correctly on
   2026-08-09, called ``sys.exit(1)``, and then sat in
   ``Py_Finalize -> wait_for_thread_shutdown -> ThreadHandle_join`` for three
   days waiting on an orphaned JSOC socket read. It held the run lock the
   whole time, so the next three nightly runs exited as "already running"
   and 4k backfill silently stopped.

A daemon thread has neither property: it is never joined at shutdown, so an
orphaned one cannot keep the process alive.

Tradeoff: on timeout the worker is deliberately abandoned (Python cannot kill
threads). It keeps running in the background until the underlying network
call finishes or errors out on its own, holding its socket and memory until
then — and if the process exits first, it is torn down mid-call. That is
accepted: an abandoned thread is better than a hung monitoring loop, and a
torn-down one is better than a cron job that never exits.
"""

import threading
from concurrent.futures import TimeoutError as FutureTimeoutError

__all__ = ["run_with_timeout", "FutureTimeoutError"]

# Sentinel: distinguishes "fn returned None" from "no result recorded".
_UNSET = object()


def run_with_timeout(fn, timeout: float, label: str = ""):
    """
    Run ``fn()`` in a daemon worker thread, waiting at most ``timeout`` seconds.

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
    box = {"result": _UNSET, "error": _UNSET}
    finished = threading.Event()

    def _runner():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc
        finally:
            finished.set()

    worker = threading.Thread(
        target=_runner,
        name=f"timeout-{label}" if label else "timeout",
        daemon=True,
    )
    worker.start()

    if not finished.wait(timeout):
        # Do NOT join the hung worker — abandon it (see module docstring).
        raise FutureTimeoutError(
            f"{label or 'call'} exceeded {timeout}s timeout"
        )

    if box["error"] is not _UNSET:
        raise box["error"]
    return box["result"]
