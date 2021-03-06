import torch
from config import config

def save_checkpoint(
    epoch: int,
    epochs_since_improvement: int,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    bleu4: float,
    is_best: bool
) -> None:
    """
    Save model checkpoint.

    Args:
        epoch (int): epoch number the current checkpoint have been trained for
        epochs_since_improvement (int): number of epochs since last improvement in BLEU-4 score
        encoder (nn.Module): encoder model
        decoder (nn.Module): decoder model
        encoder_optimizer (nn.Optimizer): optimizer to update encoder's weights, if fine-tuning
        decoder_optimizer (nn.Optimizer): optimizer to update decoder's weights
        bleu4 (float): validation BLEU-4 score for this epoch
        is_best (bool): is this checkpoint the best so far?
    """

    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer
    }
    filename = 'checkpoint_' + config.aspects[config.current_aspect]['model_basename'] + '.pth.tar'
    torch.save(state, config.model_path + filename)

    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, config.model_path + 'best_' + filename)


def load_checkpoint(
    checkpoint_path: str, fine_tune_encoder: bool, encoder_lr: float
) -> tuple:
    """
    Load a checkpoint, so that we can continue to train on it.

    Args:
        checkpoint_path (str): path of the checkpoint
        fine_tune_encoder (bool): fine-tune encoder or not
        encoder_lr (float): learning rate of encoder (if fine-tune)

    Returns:
        encoder (nn.Module): Encoder model
        decoder (nn.Module): Decoder model
        encoder_optimizer (nn.Optimizer): Optimizer to update encoder's weights
            ('none' if there is no optimizer for encoder in checkpoint)
        decoder_optimizer (nn.Optimizer): Optimizer to update decoder's weights
        start_epoch (int): We should start training the model from __th epoch
        epochs_since_improvement (int): Number of epochs since last improvement
            in BLEU-4 score
        best_bleu4 (float): BLEU-4 score of checkpoint
    """

    checkpoint = torch.load(checkpoint_path)

    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_bleu4 = checkpoint['bleu-4']

    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']

    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    if config.fine_tune_encoder is True and encoder_optimizer is None:
        encoder.CNN.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(
            params = filter(lambda p: p.requires_grad, encoder.CNN.parameters()),
            lr = encoder_lr
        )

    return encoder, encoder_optimizer, decoder, decoder_optimizer, \
            start_epoch, epochs_since_improvement, best_bleu4


def clip_gradient(optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    """
    Clip gradients computed during backpropagation to avoid explosion
    of gradients.

    Args:
        optimizer (optim.Optimizer): optimizer with the gradients to be clipped
        grad_clip (flaot): clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer, shrink_factor: float
) -> None:
    """
    Shrink learning rate by a specified factor.

    Args:
        optimizer (optim.Optimizer): optimizer whose learning rate must be shrunk
        shrink_factor (float): factor in interval (0, 1) to multiply learning rate with
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute top-k accuracy, from predicted and true labels.

    Args:
        scores (torch.Tensor): Scores from the model
        targets (torch.Tensor): Ground truth labels
        k (int): k in top-k accuracy

    Returns:
        accuracy (float): Top-k accuracy
    """
    batch_size = targets.size(0)
    # Return the indices of the top-k elements along the first dimension (along every row of a 2D Tensor), sorted
    _, ind = scores.topk(k, 1, True, True)
    # The target tensor is the same for each of the top-k predictions (words). Therefore, we need to expand it to
    # the same shape as the tensor (ind)
    # (double every label in the row --> so every row will contain k elements/k columns)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    # Sum up the correct predictions --> we will now have one value (the sum)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    # Devide by the batch_size and return the percentage
    return correct_total.item() * (100.0 / batch_size)
