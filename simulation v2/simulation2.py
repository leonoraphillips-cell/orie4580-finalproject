# simulation2.py
import numpy as np
from query2 import Query
from gpu2 import GPU
from scheduler2 import get_next_batch


MODE = "baseline"  # or "prefill_first"


def run_simulation2(
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
    gpu = GPU(*gpu_params)

    # events
    t = 0.0
    next_arrival = np.random.exponential(1 / arrival_rate)
    next_finish = np.inf

    # --- Utilization tracking ---
    util_time = 0.0
    last_t = 0.0

    # --- Metrics ---
    TTFT_list = []
    TBT_all = []
    completions = 0

    while t < T_end:

        # choose next event
        t_next = min(next_arrival, next_finish)
        t = t_next

        # utilization accumulation
        if gpu.busy:
            util_time += (t - last_t)
        last_t = t

        # --------------------------
        # ARRIVAL
        # --------------------------
        if next_arrival <= next_finish:
            L = L_dist()
            B = B_dist()
            q = Query(t, L, B)
            prefill_queue.append(q)

            next_arrival = t + np.random.exponential(1 / arrival_rate)

        # --------------------------
        # GPU FINISH
        # --------------------------
        else:
            completed_batch = gpu.finish_batch()

            for (q, tokens) in completed_batch:

                if q.stage == "prefill":
                    q.remaining_prefill = 0
                    q.stage = "decode"
                    q.TTFT = t
                    TTFT_list.append(q.TTFT)
                    decode_queue.append(q)
                    prefill_queue.remove(q)

                else:  # decode token
                    q.record_decode(t)
                    if q.remaining_decode == 1:
                        q.remaining_decode = 0
                        completions += 1
                        TBT_all.extend(q.TBT_list)
                        decode_queue.remove(q)
                    else:
                        q.remaining_decode -= 1

            next_finish = np.inf

        # --------------------------
        # DISPATCH NEW BATCH IF GPU IDLE
        # --------------------------
        if not gpu.busy:
            batch = get_next_batch(prefill_queue, decode_queue, K, mode)
            if batch:
                next_finish = gpu.start_batch(batch, t)

    utilization = util_time / T_end

    return {
        "utilization": utilization,
        "TTFT": np.array(TTFT_list),
        "TBT": np.array(TBT_all),
        "completions": completions
    }

