from typing import Tuple, Union
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNEncoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        rnn_type: str,
        bidirectional: bool = False,
        layers: int = 1,
        dropout: float = 0.
    ) -> None:
        super(RNNEncoder, self).__init__()

        if bidirectional and hidden_size % 2 != 0:
            raise ValueError('The hidden dimension must be even for bidirectional encoders')
        self.directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.layers = layers
        self.hidden_size = hidden_size // self.directions

        self.rnn_type = rnn_type
        if self.rnn_type == "gru":
            rnn_class = nn.GRU
        elif self.rnn_type == "lstm":
            rnn_class = nn.LSTM
        else:
            raise RuntimeError(f'RNN type {self.rnn_type} not implemented.')

        self.rnn = rnn_class(
            embedding_size,
            self.hidden_size,
            bidirectional = bidirectional,
            num_layers = layers,
            dropout = dropout,
        )

    def forward(
        self,
        enc_input: torch.FloatTensor,
        lengths: torch.LongTensor,
        hidden: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor]:
        sorted_lengths = sorted(lengths, reverse=True)
        is_sorted = sorted_lengths == lengths
        # is_varlen = sorted_lengths[0] != sorted_lengths[-1]
        if not is_sorted:
            true2sorted = sorted(range(len(lengths)), key=lambda x: -lengths[x])
            sorted2true = sorted(range(len(lengths)), key=lambda x: true2sorted[x])
            enc_input = torch.stack([enc_input[i, :, :] for i in true2sorted], dim=0)
            lengths = [lengths[i] for i in true2sorted]

        embeddings = enc_input
        # if is_varlen:
        embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths = lengths,
            batch_first = True
        )

        output, hidden = self.rnn(embeddings, hidden)
        if self.rnn_type == "lstm":
            hidden_list = list(hidden)
        else:
            hidden_list = [hidden]
        if self.bidirectional:
            hidden_list = [
                torch.stack([
                    torch.cat((h[2*i], h[2*i+1]), dim=1)
                    for i in range(self.layers)
                ])
                for h in hidden_list
            ]
        # if is_varlen:
        output = nn.utils.rnn.pad_packed_sequence(output)[0]
        if not is_sorted:
            hidden_list = [
                torch.stack([h[:, i, :] for i in sorted2true], dim=1)
                for h in hidden_list
            ]
            output = torch.stack([output[:, i, :] for i in sorted2true], dim=1)
        if self.rnn_type == "lstm":
            hidden = tuple(hidden_list)
        else:
            hidden = hidden_list[0]
        return hidden, output

    def initial_hidden(self, batch_size: int) -> Union[Tuple[torch.FloatTensor], torch.FloatTensor]:
        hidden = torch.zeros(
            self.layers * self.directions, batch_size, self.hidden_size,
            requires_grad = False,
            device = device
        )
        if self.rnn_type == "lstm":
            cell_state = torch.zeros(
                self.layers * self.directions, batch_size, self.hidden_size,
                requires_grad = False,
                device = device
            )
            return hidden, cell_state
        else:
            return hidden
