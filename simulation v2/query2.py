# query2.py

class Query:
    def __init__(self, arrival_time, L, B):
        self.arrival_time = arrival_time
        self.L = L
        self.B = B

        self.remaining_prefill = L
        self.remaining_decode = B
        self.stage = "prefill"

        # --- Metrics ---
        self.TTFT = None
        self.last_decode_finish = None
        self.TBT_list = []   # time between tokens

    def record_decode(self, t):
        """Record decode token completion and compute TBT."""
        if self.last_decode_finish is not None:
            self.TBT_list.append(t - self.last_decode_finish)
        self.last_decode_finish = t

    def __repr__(self):
        return f"Q(arr={self.arrival_time:.2f}, L={self.L}, B={self.B}, stage={self.stage})"
