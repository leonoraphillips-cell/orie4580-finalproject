import simpy
import random

import simpy
import random

class GPUWorker:
    """
    GPU with realistic LLM service model:
    S(b) = c + a * max(0, b - b0)
    """

    def __init__(self, env, c=45.5, a=0.30, b0=64):
        self.env = env
        self.c = c      # ms
        self.a = a      # ms per token
        self.b0 = b0    # tokens
        self.resource = simpy.Resource(env, capacity=1)

    def run_batch(self, b):
        """
        Run a batch of 'b' tokens using the service time model.
        Converts ms → seconds for simpy.
        """
        service_ms = self.c + self.a * max(0, b - self.b0)

        # Convert ms to seconds for simulation time
        service_seconds = service_ms / 1000.0

        return self.env.timeout(service_seconds)

