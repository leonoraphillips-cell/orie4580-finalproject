# batcher.py
def make_prefill_batch(prefill_queue, K):
    if not prefill_queue:
        return None
    batch = []
    for q in prefill_queue[:K]:
        tokens = q.remaining_prefill
        batch.append((q, tokens))
    return batch

def make_decode_batch(decode_queue, K):
    if not decode_queue:
        return None
    batch = []
    for q in decode_queue[:K]:
        tokens = 1  # one decode token per query
        batch.append((q, tokens))
    return batch
