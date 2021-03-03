import os
import json
from tqdm import tqdm
from typing import Tuple
from torch import nn
import torch.utils.data
import torchvision.transforms as transforms

from src.multi.utils.dataloader import AspectDataset
from src.multi.runners import batch_inference
from src.metrics import Metrics
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = config.dataset_output_path
wordmap_path = os.path.join(config.dataset_output_path, 'wordmap_' + config.aspects['all']['dataset_basename'] + '.json')

# load word map (word2ix)
with open(wordmap_path, 'r') as j:
    word_map = json.load(j)

# create ix2word map
rev_word_map = {v: k for k, v in word_map.items()}

# normalization transform
normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)


def test(beam_k: int, model: nn.Module) -> Tuple[float]:
    loader = torch.utils.data.DataLoader(
        AspectDataset(
            data_folder, config.aspects, config.embed_path, config.model_path, 'test',
            max_length = config.multi_max_length,
            transform = transforms.Compose([normalize])
        ),
        batch_size = config.multi_batch_size,
        shuffle = True,
        num_workers = config.workers,
        pin_memory = True
    )

    ground_truth = list()
    prediction = list()

    for i, (feats_batch, lengths, allcaps) in enumerate(tqdm(loader, desc="Beam size: " + str(beam_k))):
        captions, _ = batch_inference(
            config = config,
            model = model,
            desired_length_func = lambda _: _//2 + 1,
            lengths = lengths,
            features = feats_batch,
            word_map = word_map,
            multi_beam_k = beam_k
        )

        # ground truths
        for img_caps in allcaps:
            img_caps = img_caps.tolist()
            img_captions = list(map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}], img_caps))  # remove <start> and pads
            ground_truth.append(img_captions)

        # predictions
        for seq in captions:
            pred = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            prediction.append(pred)

        assert len(ground_truth) == len(prediction)

        if i > 0:
            break

    # calculate metrics
    metrics = Metrics(ground_truth, prediction, rev_word_map)
    novelty, diversity, spice = metrics.novelty, metrics.diversity, metrics.spice

    return novelty, diversity, spice


if __name__ == '__main__':
    multi_beam_k = 5
    model_path = "/Users/zou/Renovamen/Developing/Food-IAC/checkpoints/multi_lstm_attn_2021.04.05.16.28.28.977_001000.p"

    # load model
    checkpoint = torch.load(model_path, map_location=str(device))
    model = checkpoint["model"]
    model.eval()

    (n1, n4), (d1, d4), spice = test(beam_k=multi_beam_k, model=model)

    print("\nScores @ beam size of %d are:" % (multi_beam_k if multi_beam_k else 1))
    print("   Novelty-1: %.4f" % n1)
    print("   Novelty-4: %.4f" % n4)
    print("   Diversity-1: %.4f" % d1)
    print("   Diversity-4: %.4f" % d4)
    print("   SPICE: %.4f" % spice)
