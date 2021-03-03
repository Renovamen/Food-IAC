import os
import json
import h5py
import numpy as np
from collections import Counter
from random import seed, choice, sample
from tqdm import tqdm

from src.utils import load_image
from config import config

def full_word_map(min_word_freq: int, info: dict, output_folder: str) -> dict:
    karpathy_json_path = info['caption_path']
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    word_freq = Counter()
    for img in data['images']:
        for c in img['sentences']:
            word_freq.update(c['tokens'])

    # create word map (word2id)
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # create a base/root name for all output files
    base_filename = info['dataset_basename']

    # save word map (word2id) to a JSON
    with open(os.path.join(output_folder, 'wordmap_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    return word_map


def prepro_aspect(
    karpathy_json_path: str,
    image_folder: str,
    captions_per_image: int,
    base_filename: str,
    output_folder: str,
    word_map: dict,
    max_len: int = 100,
    resized_size: int = 256
) -> None:
    print('Preprocessing {} ...'.format(base_filename))

    # load Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = [] # list of images' path
    train_image_captions = [] # list of captions: [img1_cap[cap_1[token_1, .., token_n], ..., cap_n[]], ..., imgn_cap[]]

    val_image_paths = []
    val_image_captions = []

    test_image_paths = []
    test_image_captions = []

    for img in data['images']:
        captions = [] # all (validate) captions of current image
        for c in img['sentences']:
            # make sure the sentence is shorter than max_len, or ignore it
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filename'])  # path to current image

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [
        (train_image_paths, train_image_captions, 'train'),
        (val_image_paths, val_image_captions, 'val'),
        (test_image_paths, test_image_captions, 'test')
    ]:

        with h5py.File(os.path.join(output_folder, split + '_images_' + base_filename + '.hdf5'), 'a') as h:
            # make a note of the number of captions that are sampled per image
            h.attrs['captions_per_image'] = captions_per_image

            # create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, resized_size, resized_size), dtype = 'uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):
                # sample captions
                if len(imcaps[i]) < captions_per_image:
                    # if num of captions of current image is less than captions_per_image,
                    # then complete it to captions_per_image from any of the captions
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    # if num of captions is more than captions_per_image,
                    # just select captions_per_image captions randomly
                    captions = sample(imcaps[i], k = captions_per_image)

                # sanity check
                assert len(captions) == captions_per_image

                # read image
                img = load_image(impaths[i])

                # save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # one-hot encoding and pad the caption to fit max_len
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] +\
                            [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # length of caption without padding (with '<start>' and '<end>')
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # sanity check
            # images.shape[0] = len(impaths)
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_captions_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_caplength_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def prepro_data(
    aspects: dict,
    image_folder: str,
    captions_per_image: int,
    min_word_freq: int,
    output_folder: str,
    max_len: int = 100,
    resized_size: int = 256
):
    word_map = full_word_map(
        min_word_freq = min_word_freq,
        info = aspects['all'],
        output_folder = output_folder
    )
    for name in config.aspects.keys():
        prepro_aspect(
            karpathy_json_path = aspects[name]['caption_path'],
            image_folder = image_folder,
            captions_per_image = captions_per_image,
            base_filename = aspects[name]['dataset_basename'],
            output_folder = output_folder,
            word_map = word_map,
            max_len = max_len,
            resized_size = resized_size
        )


if __name__ == '__main__':
    # create input files (along with word map)
    prepro_data(
        aspects = config.aspects,
        image_folder = config.dataset_image_path,
        captions_per_image = config.captions_per_image,
        min_word_freq = config.min_word_freq,
        output_folder = config.dataset_output_path,
        max_len = config.max_caption_len,
        resized_size = 256
    )
