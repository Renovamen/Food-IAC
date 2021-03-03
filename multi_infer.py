from typing import Optional
import torch
from torch import nn

from src.utils import *
from src.multi.utils import load_nets
from src.multi.runners import batch_inference
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # paths
    model_path = "/Users/zou/Renovamen/Developing/Food-IAC/checkpoints/multi_lstm_attn_2021.04.05.16.28.28.977_001000.p"
    image_path = os.path.join(config.dataset_image_path, '171300.jpg')
    wordmap_path = os.path.join(config.dataset_output_path, 'wordmap_' + config.aspects['all']['dataset_basename'] + '.json')

    single_beam_k = 5  # beam size for single-aspect captioning
    multi_beam_k = 5  # beam size for multi-aspect captioning

    # load model
    checkpoint = torch.load(model_path, map_location=str(device))
    model = checkpoint["model"]
    model.eval()

    # load word map (word2idx)
    with open(wordmap_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # idx2word

    # load single-aspect captioning networks
    encoders, decoders = load_nets(config.model_path, config.aspects, device)

    # read and process image
    image = load_image(image_path) / 255.
    image = transform_image(image).to(device)
    image = image.unsqueeze(0)  # (1, 3, 256, 256)

    # forward each single aspect captioning model and get captions for each aspect
    for i, (encoder, decoder) in enumerate(zip(encoders, decoders)):
        encoder_out = encoder(image)
        ids, _, _, feats = decoder.beam_search(encoder_out, single_beam_k, word_map)

    length = torch.LongTensor([len(ids)]).to(device)
    feats = feats.unsqueeze(0)

    captions, _ = batch_inference(
        config = config,
        model = model,
        desired_length_func = lambda _: _//2 + 1,
        lengths = length,
        features = feats,
        word_map = word_map,
        multi_beam_k = multi_beam_k
    )

    for seq in captions:
        words = [rev_word_map[ind] for ind in seq]
        print(" ".join(words))
