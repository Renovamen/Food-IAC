import numpy as np
import torch
from torch import nn

from ..utils import operations, length_control

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_autoencode_loss(
    batch, encoder: nn.Module, decoder: nn.Module, max_length: int
):
    input_feats_batch, target_ids_batch, input_lengths, target_lengths = batch
    batch_size = len(input_lengths)

    # move to GPU, if available
    input_feats_batch = input_feats_batch.to(device)
    target_ids_batch = target_ids_batch.to(device)
    input_lengths = input_lengths.to(device)
    target_lengths = target_lengths.to(device)

    hidden = encoder.initial_hidden(batch_size)

    # length penalty
    autoencode_length_countdown, autoencode_length_target = length_control.get_fixed_length_data(
        lengths = target_lengths,
        max_length = max_length,
        batch_size = batch_size
    )

    hidden, context = encoder(
        enc_input = input_feats_batch,
        lengths = input_lengths,
        hidden = hidden
    )

    # auto-encoding always uses teacher-forcing
    autoencode_ids_batch = target_ids_batch
    autoencode_length_countdown = autoencode_length_countdown[:, :autoencode_ids_batch.shape[1]]

    output = decoder.initial_output(batch_size)

    autoencode_context_mask = operations.mask(input_lengths, device=device)

    autoencode_logprobs, hidden, *output = decoder(
        ids = autoencode_ids_batch,
        lengths = target_lengths,
        length_countdown = autoencode_length_countdown.to(device),
        hidden = hidden,
        context = context,
        context_mask = autoencode_context_mask,
        prev_output = output
    )

    autoencode_loss = operations.masked_nllloss(
        logprobs = autoencode_logprobs,
        target = target_ids_batch.transpose(0, 1).contiguous(),
        lengths = target_lengths,
        device = device
    )

    return autoencode_loss, autoencode_logprobs, autoencode_length_target
