import random

import numpy as np


def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
