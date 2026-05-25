"""Performance metrics collection for LLM and RAG responses."""

import time
from typing import Optional


class MetricsCollector:
    """Collects performance metrics for a single model invocation."""

    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self.response_time: Optional[float] = None

    def start(self) -> None:
        """Begin timing the operation."""
        self._start_time = time.perf_counter()
        self.response_time = None

    def stop(self) -> float:
        """End timing and compute the elapsed duration."""
        if self._start_time is None:
            raise RuntimeError(
                "start() must be called before stop(). "
                "Did you forget to call start()?"
            )

        elapsed = time.perf_counter() - self._start_time
        self.response_time = elapsed
        return elapsed

    @property
    def has_measurement(self) -> bool:
        """Return True if stop() has been successfully called at least once."""
        return self.response_time is not None

    def reset(self) -> None:
        """Clear any previous measurement."""
        self._start_time = None
        self.response_time = None