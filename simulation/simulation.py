import simpy
import random
from .gpu import GPUWorker
from .query import Query

# Store completed queries for metrics
completed_queries = []

MODE = "baseline"   # options: "baseline", "prefill_first"


RANDOM_SEED = 42
ARRIVAL_RATE = 0.5
NUM_JOBS = 5

def sample_L():
    return 50   # fixed number of prefill tokens

def sample_B():
    return 20   # fixed decode length

#  Prefill + Decode logic (used only in prefill_first)
def process_prefill(env, query, scheduler):
    """Schedule and wait for prefill in prefill-first mode."""
    print(f"{env.now:6.3f} - {query.id} PREFILL start (L={query.L})")

    done_event = env.event()
    scheduler.add_task("prefill", query.L, done_event)
    yield done_event

    query.prefill_done_time = env.now
    print(f"{env.now:6.3f} - {query.id} PREFILL done")


def process_decode(env, query, scheduler):
    """Schedule and wait for decode tokens in prefill-first mode."""
    for t in range(query.B):
        print(f"{env.now:6.3f} - {query.id} DECODE token {t+1}/{query.B}")

        done_event = env.event()
        scheduler.add_task("decode", 1, done_event)
        yield done_event

        # For TTFT/TBT metrics
        query.decode_token_times.append(env.now)

#  Query process: baseline vs. prefill-first
def query_process(env, query, scheduler):
    query.arrival_time = env.now
    print(f"{env.now:6.3f} - {query.id} arrives")

    done = env.event()

    # Baseline: send the entire query as one job
    if MODE == "baseline":
        scheduler.add_query(query, done)
        yield done
        query.finish_time = env.now
        completed_queries.append(query)
        print(f"{env.now:6.3f} - {query.id} finishes\n")
        return

    # Prefill-first mode
    yield from process_prefill(env, query, scheduler)
    yield from process_decode(env, query, scheduler)

    query.finish_time = env.now
    completed_queries.append(query)
    print(f"{env.now:6.3f} - {query.id} finishes\n")



def arrival_process(env, arrival_rate, gpu, num_jobs):
    """Generate queries and simulate their lifecycle."""
    for i in range(num_jobs):
        interarrival = random.expovariate(arrival_rate)
        yield env.timeout(interarrival)

        q = Query(id=f"Query_{i+1}", L_dist=sample_L, B_dist=sample_B)
        env.process(query_process(env, q, gpu))

def main():
    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    gpu = GPUWorker(env)

    # Scheduler selection
    if MODE == "baseline":
        from .scheduler import ServeToCompletionScheduler
        scheduler = ServeToCompletionScheduler(env, gpu)
    else:  # prefill-first
        from .batcher import SimpleBatcher
        from .scheduler import PrefillFirstScheduler
        batcher = SimpleBatcher(env, gpu, max_tokens=64)
        scheduler = PrefillFirstScheduler(env, batcher)


    print("Starting prefill/decode simulation...\n")
    env.process(arrival_process(env, ARRIVAL_RATE, scheduler, NUM_JOBS))
    env.run(until=100)   # any time long enough to finish all queries
    print("\nSimulation finished.\n")

    # --- Metrics ---
    if not completed_queries:
        print("No completed queries, nothing to report.")
        return

    # Throughput: #queries / (max finish time - min arrival time)
    first_arrival = min(q.arrival_time for q in completed_queries)
    last_finish = max(q.finish_time for q in completed_queries)
    total_time = last_finish - first_arrival
    throughput = len(completed_queries) / total_time if total_time > 0 else float('nan')

    # TTFT per query
    ttfts = [
        (q.prefill_done_time - q.arrival_time)
        for q in completed_queries
        if q.prefill_done_time is not None
    ]

    # TBT per query (average within query)
    tbt_per_query = []
    for q in completed_queries:
        times = q.decode_token_times
        if len(times) >= 2:
            gaps = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
            tbt_per_query.append(sum(gaps) / len(gaps))

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else float('nan')
    avg_tbt = sum(tbt_per_query) / len(tbt_per_query) if tbt_per_query else float('nan')

    print(f"Total completed queries: {len(completed_queries)}")
    print(f"Simulation horizon: {total_time:.3f} time units")
    print(f"Throughput (queries per unit time): {throughput:.3f}")
    print(f"Average TTFT: {avg_ttft:.4f}")
    print(f"Average TBT: {avg_tbt:.4f}")


if __name__ == "__main__":
    main()
