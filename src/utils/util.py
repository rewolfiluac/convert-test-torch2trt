import random
from typing import Tuple, List

import cv2
import numpy as np


def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
