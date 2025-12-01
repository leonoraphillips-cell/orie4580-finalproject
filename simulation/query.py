import random

class Query:
    """
    Represents a single LLM query.
    For now: 
      L = prompt length (prefill tokens)
      B = output budget (decode tokens)
    Also stores timing info for metrics.
    """
    def __init__(self, id, L_dist, B_dist):
        self.id = id
        self.L = L_dist()
        self.B = B_dist()

        # Metrics
        self.arrival_time = None
        self.prefill_done_time = None
        self.decode_token_times = []  # times of each output token
        self.finish_time = None
