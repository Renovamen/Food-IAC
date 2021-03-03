from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from ..utils import operations
from .attention import GlobalAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttnRNNDecoder(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,
        embedding_size: int,
        hidden_size: int,
        vocabulary_size: int,
        rnn_type: str = "lstm",
        layers: int = 1,
        dropout: float = 0.,
        input_feeding: bool = True,
        fine_tune: bool = True,
    ):
        """
        Decoder network, a stacked RNN attached with an attention network.

        Args:
            embeddings (torch.Tensor): Pre-trained word embeddings
            embedding_size (int): Size of the word embeddings, only make sense
                when ``embeddings = None``
            hidden_size (int): Size of RNN's hidden layers
            vocabulary_size (int): Number of words in vocabulary
            rnn_type (str, optional, default="lstm"): "lstm" / "gru"
            layers (int, optional, default=1): Number of RNN layers in a stacked RNN
            dropout (float, optional, default=0.): Dropout
            fine_tune (bool, optional, default=True): Fine-tune word embeddings
                or not
        """
        super(AttnRNNDecoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.attention = GlobalAttention(hidden_size, alignment_function='general')
        self.input_feeding = input_feeding
        self.input_size = (embedding_size + hidden_size if input_feeding else embedding_size) + 1

        self.rnn_type = rnn_type
        if self.rnn_type == "gru":
            self.stacked_rnn = StackedGRU(
                input_size = self.input_size,
                hidden_size = hidden_size,
                layers = layers,
                dropout = dropout
            )
        elif self.rnn_type == "lstm":
            self.stacked_rnn = StackedLSTM(
                input_size = self.input_size,
                hidden_size = hidden_size,
                layers = layers,
                dropout = dropout
            )
        else:
            raise RuntimeError(f"RNN type {self.rnn_type} not implemented")

        self.set_embeddings(embeddings, fine_tune)

        self.out = nn.Linear(hidden_size, vocabulary_size, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout(dropout)

    def set_embeddings(
        self, embeddings: torch.Tensor, fine_tune: bool = True
    ) -> None:
        """
        Set weights of embedding layer

        Args:
            embeddings (torch.Tensor): Word embeddings
            fine_tune (bool): Allow fine-tuning of embedding layer? (only
                makes sense when using pre-trained embeddings)
        """
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            self.embedding.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embedding.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(
        self,
        ids: torch.LongTensor,
        lengths: torch.LongTensor,
        length_countdown: torch.FloatTensor,
        hidden: torch.FloatTensor,
        context: torch.FloatTensor,
        context_mask: torch.ByteTensor,
        prev_output: torch.FloatTensor
    ):
        """
        Args:
            ids (torch.LongTensor): Idx of the current ground truch word, required
                by teacher forcing
            lengths (torch.LongTensor): Lengths of the input sentences
            length_countdown (torch.FloatTensor): "Length countdown" scalar for
                the input sentences, see our paper (page 4) for details
            hidden (torch.FloatTensor): Initial hidden states of RNN
            context (torch.FloatTensor): Context vector output by encoder
            context (torch.ByteTensor): See ``operation.mask`` for details
            prev_output (torch.FloatTensor): Output tensor for the decoder on
                the previous timestep
        """
        embeddings = self.embedding(ids)

        rnn_input_batch = torch.cat((embeddings, length_countdown), dim=2)

        output = prev_output
        scores = []

        for emb in rnn_input_batch.split(1, dim=1):
            if self.input_feeding:
                input = torch.cat([emb.squeeze(1), output], 1)
            else:
                input = emb.squeeze(1)
            output, hidden = self.stacked_rnn(input, hidden)
            output = self.attention(output, context, context_mask)
            output = self.dropout(output)
            score = self.logsoftmax(self.out(output))
            scores.append(score)
        return torch.stack(scores), hidden, output

    def forward_decode(
        self,
        config,
        hidden: torch.FloatTensor,
        context: torch.FloatTensor,
        input_lengths: torch.LongTensor,
        length_input: torch.FloatTensor,
        word_map: dict
    ):
        """
        Greedy search inference. When ``beam_k = 1``, it could be regarded as a
        batch (so faster) version of ``forward_beam_search()``.

        Args:
            hidden (torch.FloatTensor): Initial hidden states of RNN
            context (torch.FloatTensor): Context vector output by encoder
            input_lengths (torch.LongTensor): Lengths of the input sentences
            length_input (torch.FloatTensor): "Length countdown" scalar for the
                input sentences, see our paper (page 4) for details
            word_map (dict): Word map
        """
        batch_size = len(input_lengths)
        length_input = length_input.unsqueeze(2)

        translations = [[] for _ in range(batch_size)]
        logprobs_ls = []
        prev_words = batch_size * [word_map['<start>']]
        pending = set(range(batch_size))

        output = self.initial_output(batch_size)
        context_mask = operations.mask(input_lengths, device=device)

        output_length = 0
        while (output_length < config.multi_min_length or len(pending) > 0) and output_length < config.multi_max_length:
            var = torch.LongTensor([prev_words]).transpose(0, 1).to(device)
            logprobs, hidden, output = self(
                ids = var,
                lengths = batch_size * [1],
                length_countdown = length_input[:, output_length],
                hidden = hidden,
                context = context,
                context_mask = context_mask,
                prev_output = output
            )

            prev_words = logprobs.squeeze(0).max(dim=1)[1].detach().cpu().numpy().tolist()

            for i in pending.copy():
                if prev_words[i] == word_map['<end>']:
                    pending.discard(i)
                else:
                    translations[i].append(prev_words[i])
                    if len(translations[i]) >= config.max_ratio * input_lengths[i]:
                        pending.discard(i)

            logprobs_ls.append(logprobs)
            output_length += 1

        all_logprobs = torch.cat(logprobs_ls)
        return all_logprobs, translations

    def single_beam_search(
        self,
        config,
        hidden: torch.Tensor,
        context: torch.Tensor,
        input_lengths: torch.LongTensor,
        length_input: torch.FloatTensor,
        beam_k: int,
        word_map: dict
    ):
        """
        Do beam search inference on a single sentence.

        Args:
            config:
            hidden (torch.FloatTensor): Initial hidden states of RNN
            context (torch.FloatTensor): Context vector output by encoder
            input_lengths (torch.LongTensor): Lengths of the input sentences
            length_input (torch.FloatTensor): "Length countdown" scalar for the
                input sentences, see our paper (page 4) for details
            beam_k (int): Beam size
            word_map (dict): Word map
        """
        assert len(input_lengths) == 1  # make sure the input is a single sentence

        remaining_beam_k = beam_k
        input_lengths = input_lengths * beam_k
        large_length_input = length_input.unsqueeze(2).repeat(beam_k, 1, 1, 1)
        context = context.repeat(1, beam_k, 1)
        beam_batch_size = 1

        final_translations = []
        final_logprob_ls = []
        final_cumprobs = []
        translations = None
        logprobs_ls = None
        cum_logprobs = None
        prev_words = beam_batch_size * [word_map['<start>']]

        output = self.initial_output(beam_batch_size)
        context_mask = operations.mask(input_lengths * beam_k, device=device)

        output_length = 0

        while len(final_translations) < beam_k:
            var = torch.LongTensor([prev_words]).transpose(0, 1).to(device)
            logprobs, hidden, output = self(
                ids = var,
                lengths = beam_batch_size * [1],
                length_countdown = large_length_input[:beam_batch_size, output_length],
                hidden = hidden,
                context = context[:, :beam_batch_size],
                context_mask = context_mask,
                prev_output = output
            )
            top_logprobs, top_words = logprobs.squeeze(0).topk(remaining_beam_k, dim=1)
            if output_length == 0:
                beam_batch_size = remaining_beam_k
                prev_words = top_words.detach().cpu().numpy().tolist()[0]
                logprob = top_logprobs.view(-1).detach().cpu()
                cum_logprobs = logprob

                if self.rnn_type == "gru":
                    hidden = hidden.repeat(1, remaining_beam_k, 1),
                elif self.rnn_type == "lstm":
                    hidden = (
                        hidden[0].repeat(1, remaining_beam_k, 1),
                        hidden[1].repeat(1, remaining_beam_k, 1),
                    )
                else:
                    raise RuntimeError()
                output = output.repeat(remaining_beam_k, 1)
                translations = [[word] for word in prev_words]
                logprobs_ls = [[log_prob] for log_prob in logprob]
            else:
                candidate_cum_logprobs = top_logprobs.data.cpu() + cum_logprobs.view(-1, 1)
                flat_cum_probs = candidate_cum_logprobs.view(-1)
                beam_index = np.tile(np.arange(remaining_beam_k), (remaining_beam_k, 1)).T
                top_cum_logprobs, top_indices = flat_cum_probs.topk(remaining_beam_k)
                beam_chosen = beam_index.reshape(-1)[top_indices.numpy()]

                new_translations = []
                new_logprobs_ls = []
                new_cum_logprobs = []
                beam_pos = []
                prev_words = []
                for i in range(remaining_beam_k):
                    index = top_indices[i]
                    beam_index = beam_chosen[i]
                    word_id = top_words.view(-1)[index].item()
                    if word_id == word_map['<end>'] or output_length + 1 > config.max_ratio * input_lengths[i]:
                        remaining_beam_k -= 1
                        final_translations.append(translations[i])
                        final_logprob_ls.append(logprobs_ls[i])
                        final_cumprobs.append(top_cum_logprobs[i])
                    else:
                        new_translations.append(translations[beam_index] + [word_id])
                        new_logprobs_ls.append(logprobs_ls[beam_index] + [top_logprobs.view(-1)[index].item()])
                        new_cum_logprobs.append(top_cum_logprobs[i])
                        beam_pos.append(beam_index)
                        prev_words.append(word_id)

                translations = new_translations
                logprobs_ls = new_logprobs_ls
                cum_logprobs = torch.Tensor(new_cum_logprobs)

                beam_batch_size = remaining_beam_k
                if beam_batch_size:
                    if self.rnn_type == "gru":
                        hidden = torch.stack([hidden[:, pos] for pos in beam_pos], dim=1)
                    elif self.rnn_type == "lstm":
                        hidden = (
                            torch.stack([hidden[0][:, pos] for pos in beam_pos], dim=1),
                            torch.stack([hidden[1][:, pos] for pos in beam_pos], dim=1),
                        )
                    else:
                        raise RuntimeError()
                    output = torch.stack([output[pos] for pos in beam_pos], dim=0)
            output_length += 1

        sorted_index = np.argsort(final_cumprobs)[::-1]
        final_translations = [final_translations[i] for i in sorted_index]
        final_logprob_ls = [final_logprob_ls[i] for i in sorted_index]

        return final_logprob_ls, final_translations

    def forward_beam_search(
        self,
        config,
        hidden: torch.Tensor,
        context: torch.Tensor,
        input_lengths: torch.LongTensor,
        length_input: torch.FloatTensor,
        beam_k: int,
        word_map: dict
    ):
        """
        Do beam search inference on a batch of data (by calling ``single_beam_search()``
        ``batch_size`` times.)

        Args:
            hidden (torch.FloatTensor): Initial hidden states of RNN
            context (torch.FloatTensor): Context vector output by encoder
            input_lengths (torch.LongTensor): Lengths of the input sentences
            length_input (torch.FloatTensor): "Length countdown" scalar for the
                input sentences, see our paper (page 4) for details
            word_map (dict): Word map
        """
        batch_size = len(input_lengths)
        logprob_ls =[]
        translations = []
        for i in range(batch_size):
            if self.rnn_type == "gru":
                sub_hidden = hidden[:, i:i + 1, :]
            elif self.rnn_type == "lstm":
                sub_hidden = (
                    hidden[0][:, i:i+1, :],
                    hidden[1][:, i:i+1, :],
                )
            else:
                raise RuntimeError()
            sub_context = context[:, i:i+1, :]
            sub_input_lengths = [input_lengths[i]]
            sub_length_input = length_input[i:i+1, :, :]

            k_logprob_ls, k_translations = self.single_beam_search(
                config = config,
                hidden = sub_hidden,
                context = sub_context,
                input_lengths = sub_input_lengths,
                length_input = sub_length_input,
                beam_k = beam_k,
                word_map = word_map
            )
            translations.append(k_translations[0])
            logprob_ls.append(k_logprob_ls[0])

        return logprob_ls, translations

    def initial_output(self, batch_size: int) -> torch.FloatTensor:
        return torch.zeros(
            batch_size, self.hidden_size,
            requires_grad = False,
            device = device
        )


"""
Implementation of some stacked RNNs for input feeding decoding.
Based on OpenNMT-py, see: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/stacked_rnn.py
"""
class StackedGRU(nn.Module):
    """
    Our own implementation of stacked GRU.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(
        self, input_size: int, hidden_size: int, layers: int, dropout: float
    ) -> None:
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(
        self, input: torch.FloatTensor, hidden: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor]:
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
        h_1 = torch.stack(h_1)
        return input, h_1


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(
        self, input_size: int, hidden_size: int, layers: int, dropout: float
    ) -> None:
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.layers = nn.ModuleList()

        for i in range(layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(
        self, input: torch.FloatTensor, hidden: Tuple[torch.FloatTensor]
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor]]:
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)
