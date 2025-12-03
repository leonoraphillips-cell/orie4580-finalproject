# test_sim2.py

import numpy as np
from simulation2 import run_simulation2

def L_dist():
    return np.random.choice([16, 32, 64])

def B_dist():
    return np.random.geometric(0.2)

gpu_params = (5.0, 0.05, 32)
K = 4
arrival_rate = 0.05
T_end = 2000
SEED = 42


def summarize(result):
    util = result["util"]
    TTFT = result["TTFT"]
    TBT = result["TBT"]
    comp = result["completions"]

    return (
        f"util={util:.3f}, "
        f"mean TTFT={np.mean(TTFT):.2f}, "
        f"mean TBT={np.mean(TBT):.2f}, "
        f"completions={comp}"
    )


print("=== BASELINE ===")
res_base = run_simulation2(
    T_end=T_end,
    arrival_rate=arrival_rate,
    L_dist=L_dist,
    B_dist=B_dist,
    K=K,
    gpu_params=gpu_params,
    mode="baseline",
    seed=SEED,
)
print(summarize(res_base))


print("\n=== PREFILL FIRST ===")
res_pf = run_simulation2(
    T_end=T_end,
    arrival_rate=arrival_rate,
    L_dist=L_dist,
    B_dist=B_dist,
    K=K,
    gpu_params=gpu_params,
    mode="prefill_first",
    seed=SEED,
)
print(summarize(res_pf))

