import os
from pytorch_lightning.utilities.rank_zero import rank_zero_only

@rank_zero_only
def _check_rank_zero():
    return True

def is_rank_zero():
    """Helper to get the rank of the current process in a DDP setting"""
    rank_zero = False
    rank_zero = _check_rank_zero()
    return rank_zero
