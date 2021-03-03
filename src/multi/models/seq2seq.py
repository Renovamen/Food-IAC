import torch
from torch import nn, optim
from typing import Callable, List

class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.module_ls = [self.encoder, self.decoder]

    def forward(self):
        raise NotImplementedError()

    def initialize(self, param_init: float = 0.1) -> None:
        for param in self.parameters():
            param.data.uniform_(-param_init, param_init)

    def initialize_optimizer(self, learning_rate: float) -> Callable:
        optimizers = Optimizers()
        add_optimizer(self.encoder, [optimizers], learning_rate)
        add_optimizer(self.decoder, [optimizers], learning_rate)
        return optimizers


class Optimizers:
    def __init__(self, optimizer_ls: List[optim.Optimizer] = None) -> None:
        if optimizer_ls:
            self.optimizer_ls = optimizer_ls
        else:
            self.optimizer_ls = []

    def add_optimizer(self, optimizer: optim.Optimizer) -> None:
        self.optimizer_ls.append(optimizer)

    def step(self) -> None:
        for optimizer in self.optimizer_ls:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self.optimizer_ls:
            optimizer.zero_grad()


def add_optimizer(
    module: nn.Module, optimizers_list: List[Callable], lr: float
) -> optim.Optimizer:
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    for optimizers in optimizers_list:
        optimizers.add_optimizer(optimizer)
    return optimizer
