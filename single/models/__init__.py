from config import config
from .decoder import Decoder
from .encoder import Encoder

'''
setup a decoder

input params:
    vocab_size: size of vocabulary
    embed_dim: dimention of word embeddings
    embeddings: word embeddings
'''
def set_decoder(vocab_size, embed_dim, embeddings):
    model = Decoder(
        embed_dim = embed_dim,
        embeddings = embeddings,
        fine_tune = config.fine_tune_embeddings,
        attention_dim = config.attention_dim,
        decoder_dim = config.decoder_dim,
        vocab_size = vocab_size,
        dropout = config.dropout
    )
    return model


'''
setup an encoder

input params:
    embed_dim: dimention of word embeddings
'''
def set_encoder(embed_dim):
    model = Encoder(
        decoder_dim = config.decoder_dim,
        embed_dim = embed_dim
    )
    return model
