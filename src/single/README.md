# Single Aspect Captioning Module

PyTorch implementation of the single aspect captioning module, which is guaranteed to generate the captions and learn the feature representations of each aesthetic attribute.


&nbsp;

## Usage

### Run Experiments

First, configure hyperparameters and options in [`config.py`](config.py). Refer to this file for more information about each hyperparameters or options.

Then preprocess the images along with their captions and store them locally:

```bash
python preprocess.py
```

Run train phase:

```bash
python train.py
```

If you have enabled tensorboard (`tensorboard = True` in [`config.py`](config.py)), you can visualize the losses and accuracies during training by:

```bash
tensorboard --logdir=<your_log_dir>
```

Run test phase and compute metrics:

```bash
python test.py
```


&nbsp;

### Inference

Edit the following items in [`inference.py`](single_infer.py):

```python
model_path = 'path_to_trained_model'
wordmap_path = 'path_to_word_map'
img = 'path_to_image'
beam_size = 5 # beam size for beam search
```

Then run:

```bash
python inference.py
```


&nbsp;

## Some Notes

The `load_embeddings` method (in [`utils/embedding.py`](utils/embedding.py)) would create a cache under folder `dataset_output_path`, so that it could load the embeddings quicker the next time.


&nbsp;

## Acknowledgements

The code is based on [sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).

Implementations of metrics are under [`metrics`](metrics), which are adopted from [ruotianluo/coco-caption](https://github.com/ruotianluo/coco-caption).
