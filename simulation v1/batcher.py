import simpy

class SimpleBatcher:
    """
    A minimal batcher:
    - Collects tasks (each with a token count)
    - Dispatches them as soon as enough tasks accumulate
    - Uses a max batch size in tokens

    Later we'll add:
      - decode batching rules
      - timeouts
      - scheduling polices
      - prioritization
    """

    def __init__(self, env, gpu, max_tokens=64):
        self.env = env
        self.gpu = gpu
        self.max_tokens = max_tokens

        self.queue = []
        self.action = env.process(self.run())

    def add_task(self, task):
        """Add a (token_count, callback_event) task into the queue."""
        self.queue.append(task)

    def run(self):
        """Continuous batching loop."""
        while True:
            # Wait until there is at least one task
            if not self.queue:
                yield self.env.timeout(0.001)
                continue

            batch = []
            token_sum = 0

            # Greedy pack tasks up to token limit
            while self.queue and token_sum + self.queue[0][0] <= self.max_tokens:
                task = self.queue.pop(0)
                batch.append(task)
                token_sum += task[0]

            # Dispatch the batch
            # GPU time depends on token_sum using GPU.run_batch
            yield self.gpu.run_batch(token_sum)

            # Notify all tasks their batch is done
            for (tokens, done_event) in batch:
                done_event.succeed()

