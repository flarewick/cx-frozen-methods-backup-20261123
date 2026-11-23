"""Host-only cumulative and rolling utilization, including all startup time."""
from collections import deque

MIN_UTILIZATION = 40.0
WINDOW_SECONDS = 300.0


class UtilizationWindow:
    def __init__(self, window_seconds=WINDOW_SECONDS):
        if not 0 < window_seconds < float("inf"):raise ValueError("Invalid utilization window")
        self.window_seconds=float(window_seconds)
        self.elapsed = 0.0
        self.previous = {'device': 0.0}
        self.areas = {}
        self.intervals = deque()

    def add(self, elapsed, device, processes):
        if elapsed < self.elapsed:
            raise ValueError('Utilization clock moved backwards')
        rates = {'device': float(device), **{int(pid): float(value) for pid, value in processes.items()}}
        if any(not 0 <= value <= 100 for value in rates.values()):
            raise ValueError('Invalid GPU utilization')
        # Hold the preceding reading over each interval. Time before the first
        # reading and before a PID first appears contributes zero, never omitted.
        width = elapsed - self.elapsed
        for key, value in self.previous.items():
            self.areas[key] = self.areas.get(key, 0.0) + value * width
        self.intervals.append((self.elapsed, elapsed, self.previous))
        self.elapsed, self.previous = elapsed, rates
        cutoff = max(0.0, elapsed - self.window_seconds)
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        rolling = {}
        for begin, end, values in self.intervals:
            width = end - max(begin, cutoff)
            for key, value in values.items():
                rolling[key] = rolling.get(key, 0.0) + value * width
        denominator = max(elapsed, 1e-9)
        window = max(elapsed - cutoff, 1e-9)
        return ({key: self.areas.get(key, 0.0) / denominator for key in rates},
                {key: rolling.get(key, 0.0) / window for key in rates})
