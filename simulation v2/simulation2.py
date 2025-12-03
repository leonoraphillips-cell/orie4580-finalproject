# simulation2.py
import numpy as np
from query2 import Query
from gpu2 import GPU

# default mode if not overridden
MODE = "prefill_first"   # "baseline" or "prefill_first"


def get_next_query(prefill_queue, decode_queue, mode):
    """
    Decide which single query gets the NEXT TOKEN of GPU service.

    baseline:
        - Run-to-completion FIFO.
        - Always finish decode of head-of-line query before serving any prefill.

    prefill_first:
        - Always work on prefill tokens while any query still has prefill left.
        - Only decode when there is no prefill work waiting.
    """
    if mode == "prefill_first":
        if prefill_queue:
            return prefill_queue[0]
        if decode_queue:
            return decode_queue[0]
        return None

    elif mode == "baseline":
        # Run-to-completion: once a query reaches decode, finish it
        # before starting prefill on later arrivals.
        if decode_queue:
            return decode_queue[0]
        if prefill_queue:
            return prefill_queue[0]
        return None

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_simulation2(
    T_end,
    arrival_rate,
    L_dist,
    B_dist,
    K,              # unused now (kept for compatibility)
    gpu_params,     # (c, a, b0)
    mode=MODE,
    seed=None,
):
    """
    Event-driven sim with NO BATCHING.

    - Single GPU worker.
    - Poisson arrivals with rate `arrival_rate`.
    - For each query:
        L ~ L_dist()  (prefill tokens)
        B ~ B_dist()  (decode tokens)
    - GPU processes ONE TOKEN at a time:
        service time = a + Exp(c), with (c, a, b0) in gpu_params.
      Set a = 0 and choose c so that mean service time = 1/mu for M/M/1 validation.

    Returns:
        dict with keys:
            'util'         : time-average GPU utilization over [0, T_end]
            'TTFT'         : np.array of prefill-completion times (absolute)
            'TBT'          : np.array of per-token decode times from Query.TBT_list
            'completions'  : number of fully completed queries
    """

    if seed is not None:
        np.random.seed(seed)

    # Queues of Query objects
    prefill_queue = []   # queries still doing prefill
    decode_queue = []    # queries whose prefill is done

    # GPU
    gpu = GPU(*gpu_params)

    # Event times
    t = 0.0
    next_arrival = np.random.exponential(1.0 / arrival_rate)
    # gpu.done_time is tracked inside GPU

    # Utilization (time-average)
    util_time = 0.0
    last_t = 0.0

    # Metrics
    TTFT_list = []   # we store absolute prefill completion times t
    TBT_all = []     # will collect per-token decode times
    completions = 0

    # -----------------------------
    # MAIN EVENT LOOP
    # -----------------------------
    while True:
        # Next event is either an arrival or a GPU completion
        t_next = min(next_arrival, gpu.done_time)

        # Stop if the next event is beyond the horizon
        if t_next > T_end:
            if gpu.busy:
                util_time += (T_end - last_t)
            break

        t = t_next

        # Accumulate utilization time up to this event
        if gpu.busy:
            util_time += (t - last_t)
        last_t = t

        # -------------------------
        # ARRIVAL EVENT
        # -------------------------
        if next_arrival <= gpu.done_time:
            # Create a new query
            L = L_dist()
            B = B_dist()
            q = Query(t, L, B)
            prefill_queue.append(q)

            # Schedule next arrival
            next_arrival = t + np.random.exponential(1.0 / arrival_rate)

        # -------------------------
        # GPU COMPLETION EVENT
        # -------------------------
        else:
            # Finish ONE token of service for the current query
            q = gpu.finish_token(t)

            if q.stage == "prefill":
                q.remaining_prefill -= 1

                if q.remaining_prefill <= 0:
                    # Prefill done → move to decode
                    q.stage = "decode"
                    q.TTFT = t          # store absolute prefill completion time
                    TTFT_list.append(q.TTFT)

                    if q in prefill_queue:
                        prefill_queue.remove(q)
                    decode_queue.append(q)

            else:  # decode stage
                # Record this decode token's completion time for TBT
                q.record_decode(t)
                q.remaining_decode -= 1

                if q.remaining_decode <= 0:
                    completions += 1
                    if q in decode_queue:
                        decode_queue.remove(q)
                    # collect all per-token TBTs for this query
                    TBT_all.extend(q.TBT_list)

        # -------------------------
        # DISPATCH NEXT TOKEN IF GPU IDLE
        # -------------------------
        if not gpu.busy:
            q_next = get_next_query(prefill_queue, decode_queue, mode)
            if q_next is not None:
                gpu.start_token(q_next, t)

    # Final utilization
    utilization = util_time / T_end

    return {
        "util": utilization,
        "TTFT": np.array(TTFT_list),
        "TBT": np.array(TBT_all),
        "completions": completions,
    }
