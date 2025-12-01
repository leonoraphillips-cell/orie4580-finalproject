import simpy
import random
from .gpu import GPUWorker
from .query import Query
from .batcher import SimpleBatcher


RANDOM_SEED = 42
ARRIVAL_RATE = 0.5
SERVICE_RATE_PREFILL = 1.0    # mean rate for a token in prefill
SERVICE_RATE_DECODE = 5.0     # decode per-token is usually faster
NUM_JOBS = 5                  # fewer for clearer printout

# --- distributions for L and B ---
def sample_L():
    return random.randint(20, 200)    # prompt length

def sample_B():
    return random.randint(10, 50)     # output length

def process_prefill(env, query, batcher):
    print(f"{env.now:6.3f} - {query.id} PREFILL start (L={query.L})")

    done = env.event()
    batcher.add_task((query.L, done))
    yield done

    print(f"{env.now:6.3f} - {query.id} PREFILL done")


def process_decode(env, query, batcher):
    for t in range(query.B):
        print(f"{env.now:6.3f} - {query.id} DECODE token {t+1}/{query.B}")

        done = env.event()
        batcher.add_task((1, done))
        yield done



def query_process(env, query, gpu):
    """Full LLM query lifecycle."""
    print(f"{env.now:6.3f} - {query.id} arrives")

    # PREFILL
    yield from process_prefill(env, query, gpu)

    # DECODE
    yield from process_decode(env, query, gpu)

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
    batcher = SimpleBatcher(env, gpu, max_tokens=64)


    print("Starting prefill/decode simulation...\n")
    env.process(arrival_process(env, ARRIVAL_RATE, batcher, NUM_JOBS))
    env.run()
    print("\nSimulation finished.")

if __name__ == "__main__":
    main()
