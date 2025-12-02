# simulation.py
import numpy as np
from query2 import Query
from gpu2 import GPU
from scheduler2 import get_next_batch

MODE = "baseline"  # or "prefill_first"

def run_simulation(
    T_end,
    arrival_rate,
    L_dist,
    B_dist,
    K,
    gpu_params,
    mode=MODE,
    seed=None
):

    if seed is not None:
        np.random.seed(seed)

    # queues
    prefill_queue = []
    decode_queue = []

    # GPU
    gpu = GPU(*gpu_params)  # (c, a, b0)

    # events
    t = 0.0
    next_arrival = np.random.exponential(1 / arrival_rate)
    next_finish = np.inf

    # stats
    TTFT = []
    departures = 0

    while t < T_end:

        # choose next event
        t_next = min(next_arrival, next_finish)
        t = t_next

        # --------------------------
        # ARRIVAL EVENT
        # --------------------------
        if next_arrival <= next_finish:
            L = L_dist()
            B = B_dist()
            q = Query(t, L, B)
            prefill_queue.append(q)

            next_arrival = t + np.random.exponential(1 / arrival_rate)

        # --------------------------
        # GPU FINISH EVENT
        # --------------------------
        else:
            completed_batch = gpu.finish_batch()

            # update queries
            for (q, tokens) in completed_batch:
                if q.stage == "prefill":
                    q.remaining_prefill = 0
                    q.stage = "decode"
                    q.TTFT = t
                    decode_queue.append(q)
                    prefill_queue.remove(q)
                else:
                    q.remaining_decode -= 1
                    if q.remaining_decode == 0:
                        departures += 1
                        decode_queue.remove(q)

            next_finish = np.inf

        # --------------------------
        # GPU DISPATCH
        # --------------------------
        if not gpu.busy:
            batch = get_next_batch(prefill_queue, decode_queue, K, mode)
            if batch:
                next_finish = gpu.start_batch(batch, t)

    return {
        "TTFT": np.array([q.TTFT for q in decode_queue if q.TTFT is not None]),
        "departures": departures
    }
