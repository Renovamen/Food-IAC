"""Define the hyper parameters here."""

import os
from collections import OrderedDict

class config:
    # --------- general parameters ---------
    base_path = os.path.abspath(os.path.dirname(__file__))  # path to this project
    tensorboard = True  # enable tensorboard or not?

    # dataset parameters
    dataset_image_path = os.path.join(base_path, 'data/images/')
    dataset_output_path = os.path.join(base_path, 'outputs/')  # folder with data files saved by preprocess.py

    # checkpoints
    model_path = os.path.join(base_path, 'checkpoints/')  # path to save checkpoints

    # word embeddings path
    embed_path = os.path.join(base_path, 'data/glove/glove.6B.300d.txt')

    # dataloader
    workers = 0  # `num_workers` in dataloader

    # preprocess parameters
    captions_per_image = 5
    min_word_freq = 0  # words with frenquence lower than this value will be mapped to '<UNK>'
    max_caption_len = 50  # captions with length higher than this value will be ignored,
                          # with length lower than this value will be padded from right side to fit this length

    # folder to save logs output by tensorboard, only makes sense when `tensorboard = True`
    single_log_dir = os.path.join(base_path, 'logs/single/')

    # --------- single aspect captioning model ---------

    # four aspects
    aspects = OrderedDict()
    aspects['general_impression'] = {
        'caption_path': os.path.join(base_path, 'data/final/general_impression.json'),
        'dataset_basename': 'fiac_general',  # any name you want
        'model_basename': 'single_general'  # any name you want
    }
    aspects['color_light'] = {
        'caption_path': os.path.join(base_path, 'data/final/color_light.json'),
        'dataset_basename': 'fiac_color',
        'model_basename': 'single_color'
    }
    aspects['composition'] = {
        'caption_path': os.path.join(base_path, 'data/final/composition.json'),
        'dataset_basename': 'fiac_composition',
        'model_basename': 'single_composition'
    }
    aspects['dof_and_focus'] = {
        'caption_path': os.path.join(base_path, 'data/final/dof_and_focus.json'),
        'dataset_basename': 'dof',
        'model_basename': 'single_dof'
    }
    # full dataset
    aspects['all'] = {
        'caption_path': os.path.join(base_path, 'data/final/all.json'),
        'dataset_basename': 'fiac_all'
    }

    current_aspect = 'general_impression'

    # model parameters
    single_attention_dim = 512  # dimension of attention network
    single_decoder_dim = 300  # dimension of decoder's hidden layer
    single_dropout = 0.5

    # training parameters
    single_epochs = 20
    single_batch_size = 80
    single_fine_tune_encoder = False  # fine-tune encoder or not
    single_encoder_lr = 1e-4  # learning rate for encoder (if fine-tune)
    single_decoder_lr = 4e-4  # learning rate for decoder
    single_grad_clip = 5.  # gradient threshold in clip gradients
    single_checkpoint = None  # path to load checkpoint, None if none
    single_fine_tune_embeddings = True  # fine-tune word embeddings or not

    # --------- multi aspect captioning model ---------

    multi_model_basename = 'multi_lstm_attn'  # any name you want
    multi_log_dir = os.path.join(base_path, 'logs/multi/')

    model_save_every = 5
    use_gpu = False  # use GPU or not

    # dataset parameters
    multi_min_length = 2
    multi_max_length = 100

    # training parameters
    multi_epochs = 20
    multi_lr = 0.0005
    multi_lr_mult = 0.9
    multi_lr_mult_every = 10000
    multi_batch_size = 64
    multi_grad_clip = 2

    # loss multipliers
    length_penalty_multiplier = 0.0
    autoencode_loss_multiplier = 1.0

    # model parameters (general)
    multi_hidden_size = 512
    multi_encoder_birectional = True
    multi_encoder_nlayers = 3
    multi_decoder_nlayers = 3
    multi_rnn_type = "lstm"  # "gru" / "lstm"
    multi_encoder_dropout = 0.0
    multi_decoder_dropout = 0.0
    max_ratio = 1.5

    # model parameters (DAE)
    ae_noising = True
    ae_add_noise_perc_per_sent_low = 0.4
    ae_add_noise_perc_per_sent_high = 0.6
    ae_add_noise_num_sent = 2
