import numpy as np


def get_dummy_input_np(
    batch_size: int,
    color_size: int,
    hetigh_size: int,
    width_size: int,
) -> np.ndarray:
    input_shape_size = (
        batch_size,
        color_size,
        hetigh_size,
        width_size,
    )
    input_data = np.random.randint(0, 255, input_shape_size)
    return input_data
