from typing import Optional, Callable
import torch
from torch import nn

from src.multi.utils import length_control

def batch_inference(
    config,
    model: nn.Module,
    desired_length_func: Callable,
    lengths: torch.LongTensor,
    features: torch.FloatTensor,
    word_map: dict,
    multi_beam_k: Optional[int] = None
):
    batch_size = len(lengths)
    desired_lengths = desired_length_func(lengths)

    with torch.no_grad():
        hidden = model.encoder.initial_hidden(batch_size)
        length_input, length_target = length_control.get_fixed_length_data(
            lengths = desired_lengths,
            max_length = config.multi_max_length + 2,
            batch_size = batch_size
        )

        hidden, context = model.encoder(
            enc_input = features,
            lengths = lengths,
            hidden = hidden
        )

        if multi_beam_k is None:
            all_logprobs, translations = model.decoder.forward_decode(
                config = config,
                hidden = hidden,
                context = context,
                input_lengths = lengths,
                length_input = length_input,
                word_map = word_map
            )
        else:
            all_logprobs, translations = model.decoder.forward_beam_search(
                config = config,
                hidden = hidden,
                context = context,
                input_lengths = lengths,
                length_input = length_input,
                word_map = word_map,
                beam_k = multi_beam_k
            )

    return translations, all_logprobs
