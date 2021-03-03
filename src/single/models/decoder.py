from typing import Tuple
import torch
from torch import nn
import torchvision
from torch.nn import init
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sentinel(nn.Module):
    """
    Compute visual sentinel s_t.

    Args:
        input_size (int): Dimention of x_t ([ w_t; v^g ] => 2 * embed_dim)
        decoder_dim (int): Dimention of decoder's hidden layer
        dropout (float): Dropout
    """

    def __init__(
        self, input_size: int, decoder_dim: int, dropout: float = 0.5
    ) -> None:
        super(Sentinel, self).__init__()
        self.w_x = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(input_size, decoder_dim)
        )  # W_x
        self.w_h = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(decoder_dim, decoder_dim)
        )  # W_h
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self) -> None:
        """
        Intalize W_x and W_h.
        """
        init.xavier_uniform_(self.w_x[1].weight)
        init.xavier_uniform_(self.w_h[1].weight)

    def forward(
        self, x_t: torch.Tensor, h_last: torch.Tensor, m_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t (torch.Tensor): Input tensor ([ w_t; v^g ]) (batch_size, 2 * embed_dim)
            h_last (torch.Tensor): Hidden state at time ``t-1`` (batch_size, decoder_dim)
            m_t (torch.Tensor): Cell state at time ``t`` (batch_size, decoder_dim)

        Returns:
            s_t (torch.Tensor): Visual sentinel (batch_size, decoder_dim)
        """
        # eq.9: g_t = sigmoid(W_x * x_t + W_h * h_{t-1})
        g_t = self.w_x(x_t) + self.w_h(h_last)  # (batch_size, decoder_dim)
        g_t = self.sigmoid(g_t)  # (batch_size, decoder_dim)
        # eq.10: s_t = g_t ⊙ tanh(m_t)
        s_t = g_t * self.tanh(m_t)  # (batch_size, decoder_dim)
        return s_t


class LSTM1(nn.Module):
    """
    LSTM-1

    Args:
        input_size (int): Dimention of x_t^1 ([ h_last^2; w_t; v^g ] => 2 * embed_dim + decoder_dim)
        decoder_dim (int): Dimention of decoder's hidden layer
    """

    def __init__(self, input_size: int, decoder_dim: int) -> None:
        super(LSTM1, self).__init__()
        self.LSTM = nn.LSTMCell(input_size, decoder_dim, bias = True)
        self.sentinel = Sentinel(input_size, decoder_dim)

    def forward(
        self, x_t: torch.Tensor, states: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            x_t (torch.Tensor): Input tensor ([ h_last^2; w_t; v^g ]) (batch_size, 2 * embed_dim + decoder_dim)
            states (tuple): A tuple:
                0. h_last^1 (torch.Tensor): hidden state of LSTM-1 at time t-1 (batch_size, decoder_dim)
                1. m_last^1 (torch.Tensor): cell state of LSTM-1 at time t-1 (batch_size, decoder_dim)

        Returns:
            h_t^1 (torch.Tensor): Hidden state of LSTM-1 at time ``t`` (batch_size, decoder_dim)
            m_t^1 (torch.Tensor): Cell state of LSTM1 at time ``t`` (batch_size, decoder_dim)
            s_t (torch.Tensor): Visual sentinel at time ``t`` (batch_size, decoder_dim)
        """
        h_last, m_last = states
        h_t, m_t = self.LSTM(x_t, (h_last, m_last))
        s_t = self.sentinel(x_t, h_last, m_t)
        return h_t, m_t, s_t


class LSTM2(nn.Module):
    """
    LSTM-2

    input params:
        input_size (int): Dimention of x_t^2 ([ c_t; h_t^1 ] => 2 * decoder_dim)
        decoder_dim (int): Dimention of decoder's hidden layer
    """

    def __init__(self, input_size: int, decoder_dim: int) -> None:
        super(LSTM2, self).__init__()
        self.LSTM = nn.LSTMCell(input_size, decoder_dim, bias = True)

    def forward(
        self, x_t: torch.Tensor, states: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            x_t (torch.Tensor): Input tensor ([ c_t; h_t^1 ]) (batch_size, 2 * decoder_dim)
            states (tuple): A tuple:
                0. h_last^2 (torch.Tensor): Hidden state of LSTM-2 at time ``t-1`` (batch_size, decoder_dim)
                1. m_last^2 (torch.Tensor): Cell state of LSTM-2 at time ``t-1`` (batch_size, decoder_dim)

        Returns:
            h_t^2 (torch.Tensor): Hidden state of LSTM-2 at time ``t`` (batch_size, decoder_dim)
            m_t^2 (torch.Tensor): Cell state at of LSTM-2 time ``t`` (batch_size, decoder_dim)
        """
        h_last, m_last = states
        h_t, m_t = self.LSTM(x_t, (h_last, m_last))
        return h_t, m_t


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention

    Args:
        attention_dim (int): Dimention of attention network
        decoder_dim (int): Dimention of decoder's hidden layer
        dropout (float): Dropout
    """

    def __init__(
        self, attention_dim: int, decoder_dim: int, dropout: float = 0.5
    ) -> None:
        super(AdaptiveAttention, self).__init__()

        self.affine_s = nn.Linear(decoder_dim, decoder_dim)
        self.affine_h = nn.Linear(decoder_dim, decoder_dim)

        self.w_s = nn.Linear(decoder_dim, attention_dim)
        self.w_g = nn.Linear(decoder_dim, attention_dim)
        self.w_v = nn.Linear(decoder_dim, attention_dim)
        self.w_h = nn.Linear(attention_dim, 1)
        self.W_p = nn.Linear(decoder_dim, decoder_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

    def forward(
        self, V: torch.Tensor, h_t: torch.Tensor, s_t: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            V (torch.Tensor): Spatial image feature, V = [ v_1, v_2, ... v_num_pixels ] (batch_size, num_pixels = 49, decoder_dim)
            h_t (torch.Tensor): Hidden state at time t (batch_size, decoder_dim)
            s_t (torch.Tensor): Visual sentinel at time t (batch_size, decoder_dim)

        Returns:
            adaptive_out (torch.Tensor): Context vector in adaptive attention (batch_size, decoder_dim)
            alpha_hat_t (torch.Tensor): Attention weight in adaptive attention (batch_size, num_pixels)
            beta_t (torch.Tensor): Sentinel gate in adaptive attention (batch_size, 1)
        """
        s_t = self.relu(self.affine_s(s_t))  # (batch_size, decoder_dim)
        h_t = self.tanh(self.affine_h(h_t))  # (batch_size, decoder_dim)

        # W_g * h_t * 1^T
        hidden_att = self.w_g(h_t).unsqueeze(1)  # (batch_size, 1, attention_dim)
        # W_v * V
        visual_att = self.w_v(V)  # (batch_size, num_pixels, attention_dim)
        # W_s * s_t
        sentinel_att = self.w_s(s_t) # (batch_size, attention_dim)

        # [W_v * V; W_s * s_t]
        extended = torch.cat([visual_att, sentinel_att.unsqueeze(1)], dim = 1)  # (batch_size, num_pixels + 1, attention_dim)
        #   [z_t; w_h * tanh(W_s * s_t + W_g * h_t)]
        # = [w_h * tanh(W_v * V + W_g * h_t * 1^T); w_h * tanh(W_s * s_t + W_g * h_t)]
        # = w_h * tanh([W_v * V; W_s * s_t] + W_g * [h_t * 1^T; h_t])
        extended = self.tanh(extended + hidden_att)  # (batch_size, num_pixels + 1, attention_dim)
        z_t_extended = self.w_h(extended).squeeze(2)  # (batch_size, num_pixels + 1)
        # eq.12: \hat{α}_t
        # = softmax([z_t; w_h * tanh(W_s * s_t + W_g * h_t)])
        # = softmax(z_t_extended)
        alpha_hat_t = self.softmax(z_t_extended)  # (batch_size, num_pixels + 1)

        concat_feature = torch.cat([V, s_t.unsqueeze(1)], dim = 1)  # (batch_size, num_pixels + 1, decoder_dim)
        c_hat_t = (concat_feature * alpha_hat_t.unsqueeze(2)).sum(dim = 1)  # (batch_size, decoder_dim)

        # eq.13: W_p(c_hat_t + h_t)
        adaptive_out = self.tanh(self.W_p(c_hat_t + h_t))

        # β_t = \hat{α}_t[k + 1]
        beta_t = alpha_hat_t[:, -1].unsqueeze(1)  # (batch_size, 1)

        # remember, \hat{α} has been extended to num_pixels + 1 in adaptive attention, and we don't need the last element (used to compute beta) anymore
        return adaptive_out, alpha_hat_t[:, :-1], beta_t


class Decoder(nn.Module):
    """
    Decoder with two-layer LSTM and adaptive attention

    Args:
        embed_dim (int): Dimention of word embeddings
        embeddings (torch.Tensor): Word embeddings
        fine_tune (bool): Allow fine-tuning of embedding layer? (only makes
            sense when using pre-trained embeddings)
        attention_dim (int): Dimention of attention network
        decoder_dim (int): Dimention of decoder's hidden layer
        vocab_size (int): Size of vocab vocabulary
        dropout (float): Dropout
    """

    def __init__(
        self,
        embed_dim: int,
        embeddings: torch.Tensor,
        fine_tune: bool,
        attention_dim: int,
        decoder_dim: int,
        vocab_size: int,
        dropout: float = 0.5
    ) -> None:
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.set_embeddings(embeddings, fine_tune)

        self.dropout = nn.Dropout(p = self.dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)  # layer to calc word probability over vocabulary
        self.init_weights()  # initialize embedding and fc layer with the uniform distribution

        # input is word embedding concatenated with global image feature and
        # hidden state of LSTM2 at the last timestep,
        # so the size of input should be embed_dim * 2 + decoder_dim ([ h_last^2; w_t; v^g ] => embed_dim * 2 + decoder_dim)
        self.LSTM1 = LSTM1(embed_dim * 2 + decoder_dim, decoder_dim)
        # input is context vector concatenated with hidden state of LSTM1 at the current timestep
        # so the size of input should be decoder_dim * 2 ([ c_t; h_t^1 ] => decoder_dim * 2)
        self.LSTM2 = LSTM2(decoder_dim * 2, decoder_dim)
        self.adaptive_attention = AdaptiveAttention(attention_dim, decoder_dim)

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
            self.embedding.weight = nn.Parameter(embeddings, requires_grad = fine_tune)

    def init_weights(self) -> None:
        """
        Initialize embedding and fc layer with the uniform distribution, bias = 0
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(
        self, spatial_feature: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Initialize cell state and hidden state for LSTM (a vector filled with 0)

        Args:
            spatial_feature (torch.Tensor): Spatial image feature (batch_size, num_pixels, decoder_dim)

        Returns:
            h (torch.Tensor): Intial hidden state (batch_size, decoder_dim)
            c (torch.Tensor): Intial cell state (batch_size, decoder_dim)
        """
        h = torch.zeros(spatial_feature.size(0), self.decoder_dim).to(device) # h_0: (batch_size, decoder_dim)
        c = torch.zeros(spatial_feature.size(0), self.decoder_dim).to(device) # c_0: (batch_size, decoder_dim)
        return h, c

    def forward(
        self,
        encoder_out: Tuple[torch.Tensor],
        encoded_captions: torch.LongTensor,
        caption_lengths: torch.LongTensor
    ) -> (torch.Tensor, torch.LongTensor, list, torch.LongTensor):
        """
        Args:
            encoder_out (tuple): A tuple:
                0. spatial_feature (torch.Tensor): Spatial image feature (batch_size, num_pixels, decoder_dim)
                1. global_image (torch.Tensor): Global image feature (batch_size, embed_dim)
            encoded_captions (torch.LongTensor): caption after one-hot encoding (batch_size, max_caption_length)
            caption_lengths (torch.LongTensor): caption length (batch_size, 1)

        return:
            predictions (torch.Tensor): word probability over vocabulary predicted by model
            encoded_captions (torch.LongTensor): sorted encoded captions
            decode_lengths (list): actual caption length - 1
            sort_ind (torch.LongTensor): sorted indices
        """
        spatial_feature, global_image = encoder_out

        batch_size = spatial_feature.size(0)
        num_pixels = spatial_feature.size(1)
        vocab_size = self.vocab_size

        # sort input captions by decreasing lengths
        # because in 'train.py', 'pack_padded_sequence' will be used to deal with the pads in captions
        # and 'pack_padded_sequence' requires the captions sorted by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # sort_ind contains elements of the batch index of the tensor encoder_out.
        # for example, if sort_ind is [3,2,0],
        # then that means the descending order starts with batch number 3,then batch number 2, and finally batch number 0.
        spatial_feature = spatial_feature[sort_ind]
        global_image = global_image[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # word embedding
        # each batch contains a caption, all batches have the same number of rows (words),
        # since we previously padded the ones shorter than max_caption_length
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # initialize hidden state and cell state
        h1, c1 = self.init_hidden_state(spatial_feature)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(spatial_feature)  # (batch_size, decoder_dim)

        # we won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # decode lengths = actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)

        global_image = global_image.unsqueeze(1).expand_as(embeddings)
        # [ w_t; v^g ]: concatenate word embeddings and global image features
        inputs = torch.cat((embeddings, global_image), dim = 2)  # (batch_size, max_caption_length, embed_dim * 2)

        # start decoding
        for t in range(max(decode_lengths)):
            # create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep.
            # note that we cannot use 'pack_padded_sequence' provided by torch.util because we are using an LSTMCell, and not an LSTM
            batch_size_t = sum([l > t for l in decode_lengths])

            # input of LSTM1 ([ h_last^2; w_t; v^g ])
            x_t_1 = torch.cat((h2[:batch_size_t], inputs[:batch_size_t, t, :]), dim = 1)  # (batch_size_t, embed_dim * 2 + decoder_dim)

            # LSTM1
            h1, c1, s = self.LSTM1(x_t_1, (h1[:batch_size_t], c1[:batch_size_t]))  # (batch_size_t, decoder_dim)

            # adaptive attention network
            att_output, _, _ = self.adaptive_attention(spatial_feature[:batch_size_t], h1, s)  # (batch_size_t, decoder_dim)

            # input of LSTM2 ([ c_t; h_t^1 ])
            x_t_2 = torch.cat((att_output, h1), dim = 1)  # (batch_size_t, decoder_dim * 2)

            # LSTM2
            h2, c2 = self.LSTM2(x_t_2, (h2[:batch_size_t], c2[:batch_size_t]))  # (batch_size_t, decoder_dim)

            # calc word probability over vocabulary
            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind

    def beam_search(
        self, encoder_out: Tuple[torch.Tensor], beam_size: int, word_map: dict
    ) -> Tuple[list]:
        """
        Beam search (used in evaluation without Teacher Forcing and inference).

        TODO: Batched beam search
        Therefore, DO NOT use a batch_size greater than 1 - IMPORTANT!

        Args:
            encoder_out (tuple): A tuple:
                0. spatial_feature: spatial image feature (1, num_pixels, decoder_dim)
                1. global_image: global image feature (1, embed_dim)
            beam_size (int): beam size
            word_map (dict): word2id map

        Returns:
            seq (list[word_id1, ..., word_idn]): Predicted sentence after beam search
            alphas (list): Attention weights at each time step
            betas (list): Sentinel gate at each time step
            hiddens (list): Hidden states at each time step
        """

        import math

        k = beam_size

        spatial_feature, global_image = encoder_out # (1, num_pixels, decoder_dim), (1, embed_dim)

        num_pixels = spatial_feature.size(1)
        enc_image_size = int(math.sqrt(num_pixels)) # enc_image_size * enc_image_size = num_pixels

        # dimention of spatial image feature should be the same as dimention of decoder's hidden layer
        decoder_dim = spatial_feature.size(-1)
        assert decoder_dim == self.decoder_dim

        # check the size of vocabulary
        assert len(word_map) == self.vocab_size
        vocab_size = len(word_map)

        # we'll treat the problem as having a batch size of k
        spatial_feature = spatial_feature.expand(k, num_pixels, decoder_dim)  # (k, num_pixels, decoder_dim)

        # tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
        # tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)
        # tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        # tensor to store top k sequences' alphas; now they're just 1
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)
        # tensor to store the top k sequences' betas; now they're just 1
        seqs_beta = torch.ones(k, 1, 1).to(device)
        # tensor to store the top k sequences' hidden states (LSTM-2); now they're just 0
        seqs_hidden = torch.zeros(k, 1, decoder_dim).to(device)

        # lists to store completed sequences, their alphas, betas and scores
        complete_seqs = list()
        complete_seqs_scores = list()
        complete_seqs_alpha = list()
        complete_seqs_beta = list()
        complete_seqs_hidden = list()

        # start decoding
        step = 1
        h1, c1 = self.init_hidden_state(spatial_feature)  # (k, decoder_dim)
        h2, c2 = self.init_hidden_state(spatial_feature)  # (k, decoder_dim)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            # input of LSTM1: [ h_last^2; w_t; v^g ]
            x1 = torch.cat((h2, torch.cat((embeddings, global_image.expand_as(embeddings)), dim = 1)), dim = 1)  # (s, embed_dim * 2 + decoder_dim)

            # LSTM1
            h1, c1, s = self.LSTM1(x1, (h1, c1))  # (s, decoder_dim)

            # adaptive attention network
            att_output, alpha, beta = self.adaptive_attention(spatial_feature, h1, s)  # (s, decoder_dim), (s, num_pixels), (s, 1)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            # input of LSTM2: ([ c_t; h_t^1 ])
            x2 = torch.cat((att_output, h1), dim = 1)  # (s, decoder_dim * 2)

            # LSTM2
            h2, c2 = self.LSTM2(x2, (h2, c2))  # (s, decoder_dim)

            # calc word probability over vocabulary
            scores = self.fc(h2)  # (batch_size, vocab_size)
            scores = F.log_softmax(scores, dim = 1)  # (s, vocab_size)

            # record score
            # (k, 1) will be expanded to (k, vocab_size), then (k, vocab_size) + (s, vocab_size) --> (s, vocab_size)
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # for the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # add new words, alphas, betas and hidden states to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim = 1) # (s, step + 1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim = 1)  # (s, step + 1, enc_image_size, enc_image_size)
            seqs_beta = torch.cat([seqs_beta[prev_word_inds], beta[prev_word_inds].unsqueeze(1)], dim = 1)  # (s, step + 1, 1)
            seqs_hidden = torch.cat([seqs_hidden[prev_word_inds], h2[prev_word_inds].detach().unsqueeze(1)], dim = 1)  # (s, step + 1, decoder_dim)

            # which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_beta.extend(seqs_beta[complete_inds])
                complete_seqs_hidden.extend(seqs_hidden[complete_inds])

            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break

            # proceed with incomplete sequences
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            seqs_beta = seqs_beta[incomplete_inds]
            seqs_hidden = seqs_hidden[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            spatial_feature = spatial_feature[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # break if things have been going on too long
            if step > 50:
                complete_seqs.extend(seqs.tolist())
                complete_seqs_scores.extend(top_k_scores)
                complete_seqs_alpha.extend(seqs_alpha.tolist())
                complete_seqs_beta.extend(seqs_beta)
                complete_seqs_hidden.extend(seqs_hidden)
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
        betas = complete_seqs_beta[i]
        hiddens = complete_seqs_hidden[i]

        return seq, alphas, betas, hiddens
