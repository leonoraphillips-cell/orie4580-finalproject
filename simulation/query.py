import random

class Query:
    """
    Represents a single LLM query.
    For now: 
      L = prompt length (prefill tokens)
      B = output budget (decode tokens)
    """
    def __init__(self, id, L_dist, B_dist):
        self.id = id
        self.L = L_dist()   # number of prefill tokens
        self.B = B_dist()   # number of decode tokens

    def __repr__(self):
        return f"Query(id={self.id}, L={self.L}, B={self.B})"
