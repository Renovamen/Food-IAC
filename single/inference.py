import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image

from utils.visual import *
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_caption(encoder, decoder, image_path, word_map, beam_size = 3):
    '''
    read an image and caption it with beam search

    input params:
        encoder: encoder model
        decoder: decoder model
        image_path: path to image
        word_map: word map
        beam_size: number of sequences to consider at each decode-step
    return:
        seq: caption
        alphas: weights for visualization
    '''

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis = 2)
    # img = imresize(img, (256, 256))
    img = np.array(Image.fromarray(img).resize((256, 256)))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    # prediction (beam search)
    seq, alphas, betas = decoder.beam_search(encoder_out, beam_size, word_map)
    return seq, alphas, betas


if __name__ == '__main__':
    model_path = os.path.join(config.model_path, 'checkpoint_single_color.pth.tar')
    img = os.path.join(config.dataset_image_path, '179146.jpg')
    wordmap_path = os.path.join(config.dataset_output_path, 'wordmap_fiac.json')
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

    # Load word map (word2ix)
    with open(wordmap_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()} # ix2word

    # encoder-decoder with beam search
    seq, alphas, betas = generate_caption(encoder, decoder, img, word_map, beam_size)
    alphas = torch.FloatTensor(alphas)
    visualize_att_beta(
        image_path = img,
        seq = seq,
        rev_word_map = rev_word_map,
        alphas = alphas,
        betas = betas,
        smooth = ifsmooth
    )
