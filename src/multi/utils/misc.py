import os
import datetime
from typing import Tuple, List
import torch

def get_datetime() -> str:
    """Return the current time in string format."""
    timestamp = datetime.datetime.now()
    return f"{timestamp:%Y.%m.%d.%H.%M.%S}." + f"{timestamp:%f}"[:3]

def load_nets(folder: str, aspects: dict, device) -> Tuple[List[torch.nn.Module]]:
    """Load single-aspect captioning networks."""

    decoders = []
    encoders = []

    for name, info in aspects.items():
        if name == 'all':
            break

        path = os.path.join(folder, 'checkpoint_' + info['model_basename'] + '.pth.tar')
        checkpoint = torch.load(path, map_location = str(device))

        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        decoders.append(decoder)

        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()
        encoders.append(encoder)

    return encoders, decoders
