import random

import torch
from torch import nn, optim

def mask(lengths, device) -> torch.ByteTensor:
    """
    In a transformer's decoder, words on the future positions should be masked
    to prevent the self-attention mechanism from "cheating" when training.

    However, in this work, a RNN would be used as the decoder, so masking operation
    is not necessary actually...
    """
    batch_size = len(lengths)
    max_length = max(lengths)
    if max_length == min(lengths):
        return None
    mask = torch.ByteTensor(batch_size, max_length).fill_(0).to(device)
    for i in range(batch_size):
        for j in range(lengths[i], max_length):
            mask[i, j] = 1
    return mask


def masked_nllloss(logprobs, target, lengths, device):
    loss_function = nn.NLLLoss(reduction='none')
    loss_raw = loss_function(
        logprobs.view(-1, logprobs.shape[-1]),
        target.view(-1)
    )
    loss_mask = torch.ones(target.shape)
    for i, length in enumerate(lengths):
        if length < loss_mask.shape[0]:
            loss_mask[length:, i] = 0
    return (loss_raw * loss_mask.view(-1).to(device)).sum() / loss_mask.sum()


def update_learning_rate(optimizer: optim.Optimizer, new_lr: float) -> None:
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = new_lr
    optimizer.load_state_dict(state_dict)


def clip_gradients(model: nn.Module, grad_clip: float) -> None:
    if grad_clip != 0.0:
        for nn_module in model.module_ls:
            nn.utils.clip_grad_norm_(nn_module.parameters(), grad_clip)
