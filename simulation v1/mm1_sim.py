import simpy
import random
import numpy as np

def mm1_sim(lambda_rate=0.5, mu=1.0, num_jobs=5000, seed=42):
    random.seed(seed)
    env = simpy.Environment()

    server = simpy.Resource(env, capacity=1)
    wait_times = []
    system_times = []

    def customer(env, name):
        arrival = env.now
        with server.request() as req:
            yield req
            wait = env.now - arrival
            wait_times.append(wait)
            service = random.expovariate(mu)
            yield env.timeout(service)
            system_times.append(env.now - arrival)

    def generator(env):
        for i in range(num_jobs):
            yield env.timeout(random.expovariate(lambda_rate))
            env.process(customer(env, i))

    env.process(generator(env))
    env.run()

    return {
        "lambda": lambda_rate,
        "mu": mu,
        "rho": lambda_rate / mu,
        "avg_wait": float(np.mean(wait_times)),
        "avg_system": float(np.mean(system_times)),
    }
