import os
import h5py
import json
import math
import random
from copy import deepcopy
from typing import Optional, Callable, Tuple, List
import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from .misc import load_nets
from src.utils import *
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def init_features(vocab_len: int, embed_dim: int) -> torch.FloatTensor:
    features = torch.FloatTensor(vocab_len, embed_dim)
    bias = np.sqrt(3.0 / features.size(1))
    torch.nn.init.uniform_(features, -bias, bias)
    return features


class AspectDataset(Dataset):
    """
    A wrap of dataset for multi-aspect captioning.

    Args:
        data_folder (str): Folder where data files are stored
        aspects (dict): Information of the aspect networks
        embed_path (str): Path to word embeddings
        model_path (str): Path to checkpoints
        split (str): 'train' / 'val' / 'test'
        beam_size (int, optiona, default=5): Beam size for beam search
        max_length (int, optional, default=100): Max length of the target sentences
        transform (optional): Image transform pipeline
    """

    def __init__(
        self,
        data_folder: str,
        aspects: dict,
        embed_path: str,
        model_path: str,
        split: str,
        beam_size: int = 5,
        max_length: int = 100,
        transform: Optional[Callable] = None
    ) -> None:
        data_name = aspects['all']['dataset_basename']

        self.split = split
        assert self.split in {'train', 'val', 'test'}

        self.h = h5py.File(os.path.join(data_folder, self.split + '_images_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']
        self.cpi = self.h.attrs['captions_per_image']  # captions per image

        if self.split:
            with open(os.path.join(data_folder, self.split + '_captions_' + data_name + '.json'), 'r') as j:
                self.captions = json.load(j)

        self.dataset_size = len(self.imgs)

        self.transform = transform
        self.beam_size = beam_size
        self.max_length = max_length

        # load single-aspect captioning models
        self.encoders, self.decoders = load_nets(model_path, aspects, device)
        # load word map
        self.word_map = load_wordmap(data_folder, data_name)
        # load word embeddings
        self.embeddings, self.embed_dim = load_embeddings(
            emb_file = embed_path,
            word_map = self.word_map,
            output_folder = data_folder,
            output_basename = data_name
        )

    def additive_noise(
        self,
        sent: list,
        ae_add_noise_perc_per_sent_low: float,
        ae_add_noise_perc_per_sent_high: float
    ) -> list:
        """
        Add noises (some words randomly sampled from the dictionary) to a given
        sentence.

        The length of the added words will be a random number between
        `ae_add_noise_perc_per_sent_low` and `ae_add_noise_perc_per_sent_high`.
        """
        assert ae_add_noise_perc_per_sent_low <= ae_add_noise_perc_per_sent_high

        sent_len = len(sent)
        min_add_len = math.floor(sent_len * ae_add_noise_perc_per_sent_low)
        max_add_len = math.ceil(sent_len * ae_add_noise_perc_per_sent_high)

        if min_add_len > self.max_length - sent_len:
            min_add_len = self.max_length - sent_len
        if max_add_len > self.max_length - sent_len:
            max_add_len = self.max_length - sent_len

        add_len = random.randint(min_add_len, max_add_len)

        add_words = random.sample(list(self.word_map.values())[:-4], add_len)
        addition = [{
            'word': word,
            'feature': self.embeddings[word, :]
        } for word in add_words]
        sent += addition
        sent = random.sample(sent, len(sent))

        return sent

    def forward_nets(
        self, image: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Forward all the singlg-aspect captioning networks and concatenate their
        outputs to get a multi-aspect caption.
        """
        image = image.to(device)
        target_caps = []

        # forward each single aspect captioning model and get captions for each aspect
        for i, (encoder, decoder) in enumerate(zip(self.encoders, self.decoders)):
            encoder_out = encoder(image)
            seq, _, _, hidden = decoder.beam_search(encoder_out, self.beam_size, self.word_map)

            for i, w in enumerate(seq):
                if i >= self.max_length:
                    break
                # strip `<start>`, `<end>` and `<pad>`
                if w not in {self.word_map['<start>'], self.word_map['<end>'], self.word_map['<pad>']}:
                    target_caps.append({
                        'word': w,
                        'feature': hidden[i, :]
                    })

        if self.split == 'train':
            target_len = len(target_caps)

            # add nosies
            input_caps = self.additive_noise(deepcopy(target_caps), 0.4, 0.6)
            # initialize input features holder
            input_feat = init_features(self.max_length + 2, self.embed_dim)
            # concatenate input features (hidden states / word embeddings)
            for i, item in enumerate(input_caps):
                input_feat[i + 1, :] = item['feature']

            input_len = torch.LongTensor([len(input_caps)])

            # pad original sentence (target)
            target_ids = [self.word_map['<start>']] \
                        + [item['word'] for item in target_caps] \
                        + [self.word_map['<end>']] \
                        + [self.word_map['<pad>']] * (self.max_length - target_len)
            target_ids = torch.LongTensor(target_ids)

            return input_feat, target_ids, input_len, torch.LongTensor([target_len])

        else:
            length = torch.LongTensor([len(seq)]).to(device)

            feats = init_features(self.max_length + 2, self.embed_dim)
            feats[1:hidden.shape[0] + 1] = hidden

            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])

            return feats, length, all_captions

    def __getitem__(
        self, i: int
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        img = img.unsqueeze(0)
        return self.forward_nets(img)

    def __len__(self) -> int:
        return self.dataset_size
