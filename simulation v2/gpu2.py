# gpu2.py (NO BATCHING VERSION)

import numpy as np

class GPU:
    """
    A single GPU server that processes ONE TOKEN at a time.
    No batching, no chunking.
    Service time = a + Exp(c), but you can set a=0 for validation.
    """

    def __init__(self, c, a, b0):
        self.c = c         # exponential rate
        self.a = a         # setup cost (use 0 for validation)
        self.b0 = b0       # unused now but kept for API compatibility

        self.busy = False
        self.current_query = None
        self.remaining_tokens = 0
        self.done_time = np.inf

        # utilization tracking
        self.total_busy = 0.0
        self.last_busy_start = None

    def start_token(self, q, now):
        """Start processing exactly ONE token from query q."""
        self.busy = True
        self.current_query = q
        self.remaining_tokens = 1

        # token service time = a + Exp(c)
        service_time = self.a + np.random.exponential(1/self.c)
        self.done_time = now + service_time

        self.last_busy_start = now

        return self.done_time

    def finish_token(self, now):
        """Finish processing ONE token."""
        self.busy = False

        # update utilization
        if self.last_busy_start is not None:
            self.total_busy += (now - self.last_busy_start)
            self.last_busy_start = None

        q = self.current_query
        self.current_query = None
        self.remaining_tokens = 0
        self.done_time = np.inf

        return q
