"""
Tests for run_with_timeout.

The behaviour that matters here is not just "returns in time" — it is that an
abandoned worker cannot outlive its usefulness and block interpreter exit.
On 2026-08-09 a backfill run aborted correctly, called sys.exit(1), and then
hung for three days in Py_Finalize joining an orphaned JSOC download thread,
holding the run lock and silently killing the next three nightly runs.
"""

import subprocess
import sys
import threading
import time

import pytest

from solar_seed.data_sources._timeout import FutureTimeoutError, run_with_timeout


class TestRunWithTimeout:
    def test_returns_result(self):
        assert run_with_timeout(lambda: 42, timeout=5) == 42

    def test_returns_none_result(self):
        """None is a real result, not "nothing happened"."""
        assert run_with_timeout(lambda: None, timeout=5) is None

    def test_reraises_exception(self):
        def boom():
            raise ValueError("kaputt")

        with pytest.raises(ValueError, match="kaputt"):
            run_with_timeout(boom, timeout=5)

    def test_raises_on_timeout(self):
        started = threading.Event()

        def slow():
            started.set()
            time.sleep(30)

        t0 = time.monotonic()
        with pytest.raises(FutureTimeoutError):
            run_with_timeout(slow, timeout=0.2, label="slow")
        elapsed = time.monotonic() - t0

        assert started.is_set()
        # Returns control immediately, does not wait for the worker
        assert elapsed < 5

    def test_worker_is_daemon(self):
        seen = {}

        def peek():
            seen['daemon'] = threading.current_thread().daemon
            seen['name'] = threading.current_thread().name

        run_with_timeout(peek, timeout=5, label="peek")
        assert seen['daemon'] is True
        assert seen['name'] == "timeout-peek"


def test_abandoned_worker_does_not_block_process_exit():
    """The regression itself: a timed-out call must not wedge shutdown."""
    script = """
import sys, time
from solar_seed.data_sources._timeout import run_with_timeout, FutureTimeoutError

try:
    run_with_timeout(lambda: time.sleep(300), timeout=0.2, label="hang")
except FutureTimeoutError:
    pass
sys.exit(1)
"""
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    elapsed = time.monotonic() - t0

    assert result.returncode == 1, result.stderr
    # Before the fix this never returned at all (the 30s timeout above would fire)
    assert elapsed < 15
