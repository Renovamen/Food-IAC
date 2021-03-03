import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_length_data_from_desired(
    desired_lengths, max_length, batch_size: int
) -> Tuple[np.ndarray]:
    length_countdown_batch = np.tile(
        np.arange(0, -max_length, -1),
        reps = (batch_size, 1)
    ) + desired_lengths.reshape(batch_size, 1)
    return (
        length_countdown_batch.astype(int),
        (length_countdown_batch == 0).astype(int)
    )

def get_fixed_length_data(lengths, max_length, batch_size: int) -> Tuple[torch.FloatTensor]:
    length_countdown, length_target = get_length_data_from_desired(
        desired_lengths = np.array(lengths),
        max_length = max_length,
        batch_size = batch_size
    )
    return (
        torch.FloatTensor(length_countdown).unsqueeze(2).to(device),
        torch.FloatTensor(length_target).to(device)
    )
