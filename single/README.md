# Single Aspect Captioning Module

PyTorch implementation of the single aspect captioning module, which is guaranteed to generate the captions and learn the feature representations of
each aesthetic attribute.


&nbsp;

## Usage

### Configuration

Configure parameters in  [`config.py`](config.py). Refer to this file for more information about each config parameter.


&nbsp;

### Preprocess

First of all, you should preprocess the images along with their captions and store them locally:

```bash
python preprocess.py
```


&nbsp;

### Pre-trained Word Embeddings

If you would like to use pre-trained word embeddings (like [GloVe](https://github.com/stanfordnlp/GloVe)), just set `embed_pretrain = True` and the path to the pre-trained vectors file (`embed_path` ) in [`config.py`](config.py). You could also choose to fine-tune or not with the `fine_tune_embeddings` parameter.

The `load_embeddings` method (in [`utils/embedding.py`](utils/embedding.py)) would create a cache under folder `dataset_output_path`, so that it could load the embeddings quicker the next time.

Or if you want to randomly initialize the embedding layer's weights, set `embed_pretrain = False` and specify the size of embedding layer (`embed_dim`).


&nbsp;

### Train

To train a model, just run:

```bash
python train.py
```

If you have enabled tensorboard (`tensorboard = True` in [`config.py`](config.py)), you can visualize the losses and accuracies during training by:

```bash
tensorboard --logdir=<your_log_dir>
```


&nbsp;

### Test

Compute evaluation metrics for a trained model on test set:

```bash
python test.py
```

Implementations of metrics are under [`metrics`](metrics), which are adopted from [ruotianluo/coco-caption](https://github.com/ruotianluo/coco-caption).


&nbsp;

### Inference

Modify the following items in [`inference.py`](inference.py):

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

## Acknowledgements

The code is based on [sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).
