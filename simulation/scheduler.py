import simpy

class ServeToCompletionScheduler:
    """
    Baseline FIFO scheduler.
    Runs a query to completion:
      - full prefill
      - full decode
    No batching, no interleaving.
    """

    def __init__(self, env, gpu):
        self.env = env
        self.gpu = gpu
        self.queue = []
        self.action = env.process(self.run())

    def add_query(self, query, done_event):
        self.queue.append((query, done_event))

    def run(self):
        while True:
            if not self.queue:
                yield self.env.timeout(0.0001)
                continue

            query, done_event = self.queue.pop(0)

            # PREFILL
            yield self.gpu.run_batch(query.L)
            query.prefill_done_time = self.env.now

            # DECODE
            for i in range(query.B):
                yield self.gpu.run_batch(1)

                # Time of each decode token
                query.decode_token_times.append(self.env.now)

            # done
            done_event.succeed()


class PrefillFirstScheduler:
    """
    Extremely simple scheduler:
    - Prefill tasks have priority over decode tasks.
    - Scheduler emits tasks to the batcher in correct order.
    """

    def __init__(self, env, batcher):
        self.env = env
        self.batcher = batcher

        self.prefill_queue = []
        self.decode_queue = []

        self.action = env.process(self.run())

    def add_task(self, task_type, token_count, done_event):
        """Add task into the correct queue."""
        if task_type == "prefill":
            self.prefill_queue.append((token_count, done_event))
        else:
            self.decode_queue.append((token_count, done_event))

    def run(self):
        """Continuously feed tasks to the batcher based on scheduling policy."""
        while True:
            # 1. If no tasks, wait a tiny bit
            if not self.prefill_queue and not self.decode_queue:
                yield self.env.timeout(0.0001)
                continue

            # 2. Choose next task
            if self.prefill_queue:
                task = self.prefill_queue.pop(0)
            else:
                task = self.decode_queue.pop(0)

            # 3. Send task to batcher
            token_count, done_event = task
            self.batcher.add_task((token_count, done_event))

            # Let batcher process; scheduler loops continuously
            yield self.env.timeout(0)
