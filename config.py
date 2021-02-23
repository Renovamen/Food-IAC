'''
Define parameters here.
'''

import os

class config:

    # global parameters
    base_path = os.path.abspath(os.path.dirname(__file__))  # path to this project

    # dataset parameters
    dataset_image_path = os.path.join(base_path, 'data/images/')
    dataset_caption_path = os.path.join(base_path, 'data/color_light.json')
    dataset_output_path = os.path.join(base_path, 'outputs/')  # folder with data files saved by preprocess.py
    dataset_basename = 'fiac'  # any name you want

    # preprocess parameters
    captions_per_image = 5
    min_word_freq = 5  # words with frenquence lower than this value will be mapped to '<UNK>'
    max_caption_len = 50  # captions with length higher than this value will be ignored,
                          # with length lower than this value will be padded from right side to fit this length

    # path to word embeddings
    embed_path = os.path.join(base_path, 'data/glove/glove.6B.300d.txt')

    # model parameters
    attention_dim = 512  # dimension of attention network
    decoder_dim = 300  # dimension of decoder's hidden layer
    dropout = 0.5

    # training parameters
    epochs = 20
    batch_size = 80
    fine_tune_encoder = False  # fine-tune encoder or not
    encoder_lr = 1e-4  # learning rate of encoder (if fine-tune)
    decoder_lr = 4e-4  # learning rate of decoder
    grad_clip = 5.  # gradient threshold in clip gradients
    checkpoint = None  # path to load checkpoint, None if none
    workers = 1  # num_workers in dataloader
    fine_tune_embeddings = True  # fine-tune word embeddings or not

    # checkpoints
    model_path = os.path.join(base_path, 'checkpoints/')  # path to save checkpoints
    single_model_basename = 'single_color'  # any name you want

    # tensorboard
    tensorboard = True  # enable tensorboard or not?
    log_dir = os.path.join(base_path, 'logs/single/')  # folder for saving logs for tensorboard
                                                       # only makes sense when `tensorboard = True`
