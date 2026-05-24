import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricsCollector:
    """
    A class to collect and calculate performance metrics for LLM calls.
    Currently tracks response time, but is designed to be easily extendable 
    for token usage and cost tracking in the future.
    """
    _start_time: Optional[float] = field(default=None, init=False, repr=False)
    _end_time: Optional[float] = field(default=None, init=False, repr=False)
    response_time: Optional[float] = field(default=None, init=False)

    def start(self) -> None:
        """Start the timer for the metric collection."""
        self._start_time = time.perf_counter()
        self._end_time = None
        self.response_time = None

    def stop(self) -> float:
        """
        Stop the timer and calculate the response time.
        
        Returns:
            The calculated response time in seconds.
        """
        if self._start_time is None:
            raise RuntimeError("MetricsCollector.start() must be called before stop()")
        
        self._end_time = time.perf_counter()
        self.response_time = self._end_time - self._start_time
        return self.response_time
