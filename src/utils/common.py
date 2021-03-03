import os
import json
import numpy as np
from PIL import Image
from imageio import imread
import torch
import torchvision.transforms as transforms

def load_wordmap(data_folder: str, data_name: str):
    """
    Load word2idx map from json.

    Args:
        data_folder (str): Folder to store dataset output files
        data_name (str): Base name of dataset
    """
    word_map_file = os.path.join(data_folder, 'wordmap_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    return word_map

def load_image(img_path: str, resized_size: int = 256):
    img = imread(img_path)

    if len(img.shape) == 2:
        # deal with grayscale image (2D)
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.array(Image.fromarray(img).resize((resized_size, resized_size)))
    img = img.transpose(2, 0, 1)

    assert img.shape == (3, resized_size, resized_size)
    assert np.max(img) <= 255

    return img

def transform_image(img) -> torch.FloatTensor:
    img = torch.FloatTensor(img)
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([normalize])
    img = transform(img)  # (3, 256, 256)
    return img
