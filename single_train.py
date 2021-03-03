from torch import nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from src.single.models import *
from src.single.training import Trainer
from src.single.utils import *
from src.utils import *

from config import config

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

def set_trainer():
    # data parameters
    data_folder = config.dataset_output_path
    data_name = config.aspects[config.current_aspect]['dataset_basename']
    word_map_name = config.aspects['all']['dataset_basename']

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load word2id map
    word_map = load_wordmap(data_folder, word_map_name)

    # create id2word map
    rev_word_map = {v: k for k, v in word_map.items()}

    # initialize encoder-decoder framework
    if config.single_checkpoint is None:
        start_epoch = 0
        epochs_since_improvement = 0
        best_bleu4 = 0.

        # ------------- word embeddings -------------
        embeddings, embed_dim = load_embeddings(
            emb_file = config.embed_path,
            word_map = word_map,
            output_folder = data_folder,
            output_basename = word_map_name
        )

        # ----------------- encoder ------------------
        encoder = Encoder(
            decoder_dim = config.single_decoder_dim,
            embed_dim = embed_dim
        )
        encoder.CNN.fine_tune(config.single_fine_tune_encoder)
        # optimizer for encoder's CNN (if fine-tune)
        if config.single_fine_tune_encoder:
            encoder_optimizer = torch.optim.Adam(
                params = filter(lambda p: p.requires_grad, encoder.CNN.parameters()),
                lr = config.single_encoder_lr
            )
        else:
            encoder_optimizer = None

        # ----------------- decoder ------------------
        decoder = Decoder(
            embed_dim = embed_dim,
            embeddings = embeddings,
            fine_tune = config.single_fine_tune_embeddings,
            attention_dim = config.single_attention_dim,
            decoder_dim = config.single_decoder_dim,
            vocab_size = len(word_map),
            dropout = config.single_dropout
        )
        # optimizer for decoder
        decoder_params = list(filter(lambda p: p.requires_grad, decoder.parameters()))
        decoder_params = decoder_params + list(encoder.global_mapping.parameters()) \
                                        + list(encoder.spatial_mapping.parameters())

        decoder_optimizer = torch.optim.Adam(
            params = decoder_params,
            lr = config.single_decoder_lr
        )

    # or load checkpoint
    else:
        encoder,
        encoder_optimizer,
        decoder,
        decoder_optimizer,
        start_epoch,
        epochs_since_improvement,
        best_bleu4 = load_checkpoint(
            checkpoint_path = config.single_checkpoint,
            fine_tune_encoder = config.single_fine_tune_encoder,
            encoder_lr = config.single_encoder_lr
        )

    # move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # loss function (cross entropy)
    loss_function = nn.CrossEntropyLoss().to(device)

    # image transform
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder, data_name, 'train',
            transform = transforms.Compose([normalize])
        ),
        batch_size = config.single_batch_size,
        shuffle = True,
        num_workers = config.workers,
        pin_memory = True
    )
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder, data_name, 'val',
            transform = transforms.Compose([normalize])
        ),
        batch_size = config.single_batch_size,
        shuffle = True,
        num_workers = config.workers,
        pin_memory = True
    )

    trainer = Trainer(
        epochs = config.single_epochs,
        device = device,
        word_map = word_map,
        rev_word_map = rev_word_map,
        start_epoch = start_epoch,
        epochs_since_improvement = epochs_since_improvement,
        best_bleu4 = best_bleu4,
        train_loader = train_loader,
        val_loader = val_loader,
        encoder = encoder,
        decoder = decoder,
        encoder_optimizer = encoder_optimizer,
        decoder_optimizer = decoder_optimizer,
        loss_function = loss_function,
        grad_clip = config.single_grad_clip,
        fine_tune_encoder = config.single_fine_tune_encoder,
        tensorboard = config.tensorboard,
        log_dir = config.single_log_dir
    )

    return trainer


if __name__ == '__main__':
    trainer = set_trainer()
    trainer.run_train()
