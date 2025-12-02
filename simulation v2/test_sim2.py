# test_sim2.py

from simulation2 import run_simulation2
import numpy as np

# Simple prompt/output distributions for demo
def L_dist():
    return np.random.choice([16, 32, 64])

def B_dist():
    return np.random.geometric(0.2)  # average ~5 tokens

gpu_params = (20, 0.3, 64)   # c=20ms, a=0.3ms/token, b0=64
K = 4                        # batch size
arrival_rate = 0.4
T_end = 2000

results = run_simulation2(
    T_end=T_end,
    arrival_rate=arrival_rate,
    L_dist=L_dist,
    B_dist=B_dist,
    K=K,
    gpu_params=gpu_params,
    mode="baseline",
    seed=42
)

print("=== RESULTS ===")
print("GPU Utilization:", results["utilization"])
print("Mean TTFT:", np.mean(results["TTFT"]))
print("Mean TBT:", np.mean(results["TBT"]))
print("Total completions:", results["completions"])
