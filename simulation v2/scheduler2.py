# scheduler2.py
from batcher2 import make_prefill_batch, make_decode_batch

class ServeToCompletionScheduler:
    """
    Baseline:
    - FIFO by query
    - Serve full prefill of one query (batched)
    - Then serve all decode tokens for that query (1 per token)
    - Never interleave across queries
    """

    def get_next_batch(self, prefill_queue, decode_queue, K):
        # If we have an active query still in prefill or decode,
        # we always continue serving THAT query only.
        if prefill_queue:
            # Serve the first query's remaining prefill only
            q = prefill_queue[0]
            return [(q, q.remaining_prefill)]
        elif decode_queue:
            # Serve the first query's decode only
            q = decode_queue[0]
            return [(q, 1)]
        return None
        
class PrefillFirstScheduler:
    """
    Prefill-first:
    - Use batching
    - Interleave queries
    - Always process PREFILL batches before DECODE batches
    """

    def get_next_batch(self, prefill_queue, decode_queue, K):
        if prefill_queue:
            return make_prefill_batch(prefill_queue, K)
        if decode_queue:
            return make_decode_batch(decode_queue, K)
        return None

