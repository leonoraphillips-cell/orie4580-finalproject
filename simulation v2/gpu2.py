# gpu.py
import numpy as np

class GPU:
    def __init__(self, c, a, b0):
        self.c = c    # setup time
        self.a = a    # per-token cost
        self.b0 = b0  # threshold where linear cost activates

        self.busy = False
        self.current_batch = None

    def batch_service_time(self, token_load):
        return self.c + self.a * max(0, token_load - self.b0)

    def start_batch(self, batch, t):
        """batch = list of (query, tokens_processed) for this iteration."""
        self.current_batch = batch
        self.busy = True

        token_load = sum(tokens for (_, tokens) in batch)
        service_time = self.batch_service_time(token_load)

        return t + service_time  # finish time

    def finish_batch(self):
        completed = self.current_batch
        self.current_batch = None
        self.busy = False

        # NEW: decrement tokens
        for (q, tokens) in completed:
            if q.stage == "prefill":
                q.remaining_prefill -= tokens
            else:
                q.remaining_decode -= tokens  # tokens = 1 for decode

        return completed

