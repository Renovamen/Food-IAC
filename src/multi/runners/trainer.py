import os
import time
from typing import Optional
import numpy as np
import torch

from ..utils import *
from ..models import autoencode
from ...utils import AverageMeter, TensorboardWriter

class Trainer:
    def __init__(
        self,
        config,
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizers: torch.optim.Optimizer,
        print_freq: int = 50,
        tensorboard: bool = False,
        log_dir: Optional[str] = None
    ) -> None:
        self.config = config
        self.print_freq = print_freq

        self.train_loader = train_loader
        self.model = model
        self.optimizers = optimizers

        self.step = 0
        self.lr = self.config.multi_lr
        self.create_time = misc.get_datetime()
        self.epoch_len = len(self.train_loader)

        self.writer = TensorboardWriter(log_dir, tensorboard)
        self.track_loss = AverageMeter(tag='loss', writer=self.writer)

    def run_train(self) -> None:
        batch_time = AverageMeter()
        data_time = AverageMeter()

        start = time.time()

        for epoch in range(1, self.config.multi_epochs + 1):
            for i, batch in enumerate(self.train_loader):
                data_time.update(time.time() - start)

                self.optimizers.zero_grad()
                self.model.train()

                self.train_step(batch)

                batch_time.update(time.time() - start)
                start = time.time()

                self.update_learning_rate()
                self.save_model(epoch, i + 1)

                step = (epoch - 1) * self.epoch_len + i
                self.writer.set_step(step=step, mode='train')

                if i % self.print_freq == 0:
                    print(
                        'Epoch: [{0}][{1}/{2}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                            epoch, i, len(self.train_loader),
                            batch_time = batch_time,
                            data_time = data_time,
                            loss = self.track_loss
                        )
                    )

    def update_learning_rate(self) -> None:
        new_lr = (
            self.config.multi_lr
            * self.config.multi_lr_mult
            ** (self.step // self.config.multi_lr_mult_every)
        )
        if new_lr != self.lr:
            for optimizer in self.optimizers.optimizer_ls:
                update_learning_rate(optimizer=optimizer, new_lr=new_lr)
            self.lr = new_lr

    def save_model(self, i_epoch: int, i_iter: int) -> None:
        step = (i_epoch - 1) * self.epoch_len + i_iter
        if step > 0 and step % self.config.model_save_every == 0:
            filename = (
                f"{self.config.multi_model_basename}_"
                f"{self.create_time}_"
                f"{step:06d}.p"
            )
            save_path = os.path.join(self.config.model_path, filename)

            torch.save({
                "optimizer": self.optimizers,
                "model": self.model
            }, save_path)

            print(f"Saved to: {save_path}")

    def train_step(self, batch) -> None:
        raw_autoencode_loss, autoencode_logprobs, target_lengths = autoencode.get_autoencode_loss(
            batch = batch,
            encoder = self.model.encoder,
            decoder = self.model.decoder,
            max_length = self.config.multi_max_length + 2  # <start> and <end>
        )

        loss = raw_autoencode_loss * self.config.autoencode_loss_multiplier
        loss.backward()

        self.track_loss.update(loss.item())

        clip_gradients(
            model = self.model,
            grad_clip = self.config.multi_grad_clip
        )
        self.optimizers.step()
