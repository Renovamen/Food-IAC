import os
import json
from typing import Tuple
import torch

from src.utils import load_image, transform_image
from src.single.utils.visual import *
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image_path: str,
    word_map: dict,
    beam_size: int = 3
) -> Tuple[list]:
    # read and process image
    image = load_image(image_path) / 255.
    image = transform_image(image).to(device)

    # encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    # prediction (beam search)
    seq, alphas, betas, _ = decoder.beam_search(encoder_out, beam_size, word_map)
    return seq, alphas, betas


if __name__ == '__main__':
    model_path = os.path.join(config.model_path, 'checkpoint_' + config.aspects[config.current_aspect]['model_basename'] + '.pth.tar')
    image_path = os.path.join(config.dataset_image_path, '171300.jpg')
    wordmap_path = os.path.join(config.dataset_output_path, 'wordmap_' + config.aspects['all']['dataset_basename'] + '.json')
    beam_size = 5
    ifsmooth = False

    # load model
    checkpoint = torch.load(model_path, map_location=str(device))

    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()

    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2idx)
    with open(wordmap_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()} # idx2word

    # encoder-decoder with beam search
    seq, alphas, betas = generate_caption(encoder, decoder, image_path, word_map, beam_size)
    alphas = torch.FloatTensor(alphas)
    visualize_att_beta(
        image_path = image_path,
        seq = seq,
        rev_word_map = rev_word_map,
        alphas = alphas,
        betas = betas,
        smooth = ifsmooth
    )
