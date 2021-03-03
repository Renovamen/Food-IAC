import sys
import torch
import torch.utils.data
import torchvision.transforms as transforms

from src.multi.utils.dataloader import AspectDataset
from src.multi.models import RNNEncoder, AttnRNNDecoder, Seq2Seq, autoencode
from src.multi.runners import Trainer
from src.utils import *

from config import config

# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_trainer(config):
    data_folder = config.dataset_output_path
    word_map_name = config.aspects['all']['dataset_basename']

    word_map = load_wordmap(data_folder, word_map_name)

    # normalization transform
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        AspectDataset(
            data_folder, config.aspects, config.embed_path, config.model_path, 'train',
            max_length = config.multi_max_length,
            transform = transforms.Compose([normalize])
        ),
        batch_size = config.multi_batch_size,
        shuffle = True,
        num_workers = config.workers,
        pin_memory = True
    )

    embeddings, embed_dim = load_embeddings(
        emb_file = config.embed_path,
        word_map = word_map,
        output_folder = data_folder,
        output_basename = word_map_name
    )

    encoder = RNNEncoder(
        embedding_size = embed_dim,
        bidirectional = config.multi_encoder_birectional,
        hidden_size = config.multi_hidden_size,
        layers = config.multi_encoder_nlayers,
        rnn_type = config.multi_rnn_type,
        dropout = config.multi_encoder_dropout,
    ).to(device)

    decoder = AttnRNNDecoder(
        embeddings = embeddings,
        embedding_size = embed_dim,
        hidden_size = config.multi_hidden_size,
        vocabulary_size = len(word_map),
        layers = config.multi_decoder_nlayers,
        rnn_type = config.multi_rnn_type,
        dropout = config.multi_decoder_dropout,
    ).to(device)

    model = Seq2Seq(
        encoder = encoder,
        decoder = decoder
    )
    model.initialize()

    optimizers = model.initialize_optimizer(
        learning_rate = config.multi_lr
    )

    trainer = Trainer(
        config = config,
        train_loader = train_loader,
        model = model,
        optimizers = optimizers,
        print_freq = 2,
        tensorboard = True,
        log_dir = config.multi_log_dir
    )

    return trainer

if __name__ == "__main__":
    trainer = set_trainer(config)
    trainer.run_train()
