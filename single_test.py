import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import Tuple

from src.single.utils.dataloader import *
from src.single.utils.common import *
from src.metrics import Metrics
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# some path
data_folder = config.dataset_output_path  # folder with data files saved by preprocess.py
data_name = config.aspects[config.current_aspect]['dataset_basename']  # base name shared by data files
word_map_name = config.aspects['all']['dataset_basename']
model_basename = config.aspects[config.current_aspect]['model_basename']

word_map_file = os.path.join(config.dataset_output_path, 'wordmap_' + word_map_name + '.json')  # word map, ensure it's the same the data was encoded with and the model was trained with
checkpoint = os.path.join(config.model_path, 'checkpoint_' + model_basename + '.pth.tar')  # model checkpoint

# load model
checkpoint = torch.load(checkpoint, map_location=str(device))

decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

# create ix2word map
rev_word_map = {v: k for k, v in word_map.items()}

# normalization transform
normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)


def test(beam_size: int) -> Tuple[float]:
    """
    Args:
        beam_size (int): Beam size, set ``beam_size = 1`` when using greedy search.

    Returns:
        scores (Tuple[float]): Metrics
    """

    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder, data_name, 'test',
            transform = transforms.Compose([normalize])
        ),
        # TODO: batched beam search
        # therefore, DO NOT use a batch_size greater than 1 - IMPORTANT!
        batch_size = 1,
        shuffle = True,
        num_workers = 1,
        pin_memory = True
    )

    # store ground truth captions and predicted captions (word id) of each image
    # for n images, each of them has one prediction and multiple ground truths (a, b, c...):
    # prediction = [ [pred1], [pred2], ..., [predn] ]
    # ground_truth = [ [ [gt1a], [gt1b], [gt1c] ], ..., [ [gtna], [gtnb] ] ]
    ground_truth = list()
    prediction = list()

    # for each image
    for i, (image, caps, caplens, allcaps) in enumerate(tqdm(loader, desc="Beam size: " + str(beam_size))):
        # move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # forward encoder
        encoder_out = encoder(image)

        # ground_truth
        img_caps = allcaps[0].tolist()
        img_captions = list(map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}], img_caps))  # remove <start> and pads
        ground_truth.append(img_captions)

        # prediction (beam search)
        seq, _, _, _ = decoder.beam_search(encoder_out, beam_size, word_map)

        pred = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        prediction.append(pred)

        assert len(ground_truth) == len(prediction)

    # calculate metrics
    metrics = Metrics(ground_truth, prediction, rev_word_map)
    scores = metrics.all_metrics

    return scores


if __name__ == '__main__':
    beam_size = 5

    (bleu1, bleu2, bleu3, bleu4), cider, rouge, meteor, spice = test(beam_size)

    print("\nScores @ beam size of %d are:" % beam_size)
    print("   BLEU-1: %.4f" % bleu1)
    print("   BLEU-2: %.4f" % bleu2)
    print("   BLEU-3: %.4f" % bleu3)
    print("   BLEU-4: %.4f" % bleu4)
    print("   CIDEr: %.4f" % cider)
    print("   ROUGE-L: %.4f" % rouge)
    print("   METEOR: %.4f" % meteor)
    print("   SPICE: %.4f" % spice)
