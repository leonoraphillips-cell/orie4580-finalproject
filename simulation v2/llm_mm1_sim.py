import numpy as np


# =========================
#   Core building blocks
# =========================

class Job(object):
    """
    A generic 'query' / job in the system.
    For the base M/M/1 model, we only need arrival time and timestamps
    to compute waiting / system times.
    Later you can extend this with tokens, stages (prefill/decode), etc.
    """
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time

        # For per-station tracking (like in HW6)
        self.timeEnterQueue = arrival_time
        self.timeStartService = arrival_time
        self.timeLeaveService = arrival_time

    def __repr__(self):
        return f"Job(arrival={self.arrival_time:.4f})"


class ServerPool(object):
    """
    General server pool: FCFS queue + c identical exponential servers.
    This is analogous to the HW6 `Queue` class.
    """
    def __init__(self, name, capacity, serviceRate):
        self.name = name
        self.capacity = capacity       # number of parallel servers (c)
        self.serviceRate = serviceRate # per-server exponential rate μ

        # State
        self.waiting = []              # jobs waiting
        self.inService = []            # jobs currently in service
        self.serversOn = 0             # how many servers currently busy

        # Stats
        self.arriveTime = []
        self.leaveTime = []
        self.queueLength = []
        self.busy = []                 # servers busy at each event
        self.timeInQueue = []
        self.timeInService = []
        self.timeInSystem = []

    def __repr__(self):
        q_str = f"{self.name} queue: {self.waiting}"
        s_str = f"{self.name} in service: {self.inService}"
        return q_str + "\n" + s_str

    # ---------- queue operations ----------

    def arrive(self, job, t):
        """Job arrives to this station."""
        self.waiting.append(job)
        job.timeEnterQueue = t
        self.arriveTime.append(t)

    def can_start_service(self):
        return self.serversOn < self.capacity and len(self.waiting) > 0

    def start_service(self, t):
        """Move one job from queue to service."""
        job = self.waiting.pop(0)
        job.timeStartService = t
        self.timeInQueue.append(t - job.timeEnterQueue)

        self.inService.append(job)
        self.serversOn += 1

    def end_service(self, t):
        """
        One server completes service on a random job in service.
        Returns the job that just finished.
        """
        idx = np.random.randint(min(self.capacity, self.serversOn))
        job = self.inService.pop(idx)

        job.timeLeaveService = t
        self.timeInService.append(t - job.timeStartService)
        self.timeInSystem.append(t - job.timeEnterQueue)
        self.leaveTime.append(t)

        self.serversOn -= 1
        return job

def time_avg_util(times, busy, T_end):
    # times: list of event times (strictly increasing)
    # busy:  list of server-count values at those times
    times_aug = [0.0] + list(times)
    busy_aug  = [0]   + list(busy)
    area = 0.0
    for i in range(len(busy_aug)-1):
        dt = times_aug[i+1] - times_aug[i]
        area += busy_aug[i] * dt
    return area / T_end


# ===================================
#   Base M/M/1 LLM-like queue
# ===================================

def run_mm1_llm_base(lambda_rate, mu_rate, t_end, seed=None):
    """
    Base-case M/M/1 simulator for your LLM project:
    - Arrivals: Poisson rate λ (each job is a 'query')
    - Service: Exponential rate μ, single GPU server (capacity=1)
    - No batching, no setup time (c=0 in your project writeup)
    This is exactly the case the project description says to
    validate against queueing theory.

    Returns:
        times: list of event times
        gpu:   ServerPool object with full stats
    """
    if seed is not None:
        np.random.seed(seed)

    # A single "GPU" server pool = M/M/1
    gpu = ServerPool("GPU", capacity=1, serviceRate=mu_rate)

    # Time
    t = 0.0

    # For CTMC-style next-event simulation (like HW6):
    # eventRates = [arrival_rate, service_rate * serversOn]
    times = []

    # initialize first event rates
    eventRates = np.array([lambda_rate, gpu.serviceRate * gpu.serversOn])
    totalRate = np.sum(eventRates)
    # if λ = 0 and no one in system, nothing will happen
    if totalRate == 0:
        return [], gpu

    cumsumRate = np.cumsum(eventRates / totalRate)
    nextEventTime = t + np.random.exponential(1 / totalRate)

    # Main loop
    while t < t_end:
        t = nextEventTime

        # pick which event occurs
        U = np.random.rand()
        event_idx = np.where(cumsumRate >= U)[0][0]
        # event_idx = 0 : arrival
        # event_idx = 1 : service completion

        if event_idx == 0:
            # Arrival at GPU
            job = Job(t)
            gpu.arrive(job, t)

            # if server idle, start service immediately
            if gpu.can_start_service():
                gpu.start_service(t)

        else:  # service completion at GPU
            # only happens if serversOn > 0
            finished_job = gpu.end_service(t)
            # In base M/M/1, job departs the system completely.
            # (You could collect finished_job somewhere if needed.)

            # see if another job can start service
            if gpu.can_start_service():
                gpu.start_service(t)

        # record state
        times.append(t)
        gpu.queueLength.append(len(gpu.waiting))
        gpu.busy.append(gpu.serversOn)

        # schedule next event
        eventRates = np.array([lambda_rate, gpu.serviceRate * gpu.serversOn])
        totalRate = np.sum(eventRates)

        if totalRate == 0:
            # No arrivals and no one in system -> stop early
            break

        cumsumRate = np.cumsum(eventRates / totalRate)
        nextEventTime = t + np.random.exponential(1 / totalRate)

    return times, gpu


# ====================================
#    Simple example / quick test
# ====================================

if __name__ == "__main__":
    lam = 0.8
    mu = 1.0
    T_end = 10_000.0 

    times, gpu = run_mm1_llm_base(lam, mu, T_end, seed=123)

    rho_theory = lam / mu
    util_time_avg = time_avg_util(times, gpu.busy, T_end)
    
    print(f"Utilization ρ (theory): {rho_theory:.3f}")
    print(f"Time-average utilization (sim): {util_time_avg:.3f}")

    print(f"Average time in system (sim): {np.mean(gpu.timeInSystem):.3f}")
    print(f"Average time in system (M/M/1 theory): {1/(mu - lam):.3f}")

