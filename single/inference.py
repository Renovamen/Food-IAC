import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
# from scipy.misc import imread, imresize
from imageio import imread
from PIL import Image
from utils.visual import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def generate_caption(encoder, decoder, image_path, word_map, beam_size = 3):

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

    model_path = '../checkpoints/checkpoint_single.pth.tar'
    img = '../data/images/1162234.jpg'
    wordmap_path = '../outputs/wordmap_fiac.json'
    beam_size = 5
    ifsmooth = False

    # load model
    checkpoint = torch.load(model_path, map_location = str(device))

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
