# query.py
class Query:
    def __init__(self, arrival_time, L, B):
        self.arrival_time = arrival_time
        self.L = L
        self.B = B
        self.remaining_prefill = L
        self.remaining_decode = B

        self.stage = "prefill"   # switches to decode after L processed
        self.TTFT = None

    def __repr__(self):
        return f"Q(arr={self.arrival_time:.2f}, L={self.L}, B={self.B}, stage={self.stage})"
