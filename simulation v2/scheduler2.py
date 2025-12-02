# scheduler.py
from batcher2 import make_prefill_batch, make_decode_batch

def get_next_batch(prefill_queue, decode_queue, K, mode):
    """
    mode:
       "baseline"      -> do batches in stage order: prefill until all done
       "prefill_first" -> aggressively prefer prefill
    """

    if mode == "prefill_first":
        if prefill_queue:
            return make_prefill_batch(prefill_queue, K)
        else:
            return make_decode_batch(decode_queue, K)

    if mode == "baseline":
        # baseline = finish all prefill for a query before decoding any others
        if prefill_queue:
            return make_prefill_batch(prefill_queue, K)
        else:
            return make_decode_batch(decode_queue, K)

    raise ValueError("Unknown scheduling mode")
