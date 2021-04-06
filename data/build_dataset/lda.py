"""
Train and save LDA models for topic classification. The output files will be
saved to `lda_path`.
"""

import json
import os
import io
from tqdm import tqdm
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import gensim
import numpy as np

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
input_path = os.path.join(base_path, 'clean.json')
lda_path = os.path.join(base_path, 'lda')


def get_all_ngrams(cap):
    unigram_list = cap['unigrams']
    bigram_list = cap['bigrams']
    ngram_list = [[unigram] for unigram in unigram_list] + bigram_list
    final_ngram_list = [' '.join([i[0] for i in ngram]) for ngram in ngram_list]
    return final_ngram_list


def create_corpus(image_list):
    text_corpus = []

    for cnt, img in enumerate(tqdm(image_list, desc = 'Processed images: ')):
        captions = img['sentences']

        # get ngrams of each caption
        ngram_list = list(map(get_all_ngrams, captions))

        # create corpus
        for ngrams in ngram_list:
            text_corpus.append(ngrams)

    return text_corpus

def create_lda_input(text_corpus):
    """
    Create idx2word dictionary and term document frequency for LDA.

    Args:
        text_corpus (List[[ngrams_of_cap_1], ..., [ngrams_of_cap_n]]): Each item
            in the list is a list of ngrams of each caption

    Returns:
        dictionary: Idx2word dictionary
        doc_term_matrix: Term document frequency
    """

    # create id2word dictionary
    dictionary = corpora.Dictionary(text_corpus)
    dictionary.filter_extremes(no_below = 30, no_above = 0.10)
    # term document frequency
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_corpus]

    print ("dictionary shape: %d \ndoc_term_matrix shape: %d" % (len(dictionary), len(doc_term_matrix)))

    return dictionary, doc_term_matrix


def lda_train(text_corpus, num_topics_start = 10, num_topics_end = 200):
    # get id2word dictionary and term document frequency
    dictionary, doc_term_matrix = create_lda_input(text_corpus)

    print("----------- LDA training start -----------")
    LDA = gensim.models.ldamulticore.LdaMulticore

    for num_topics in range(num_topics_start, num_topics_end, 5):

        print("-------- topics: ", num_topics, " --------")

        # train a LDA model of current num_topics
        ldaModel = LDA(
            corpus = doc_term_matrix,
            num_topics = num_topics,
            id2word = dictionary,
            passes = 200,
            workers = 15,
            iterations = 5000,
            chunksize = 20000
        )

        # coherence score of current model
        # lower score is better (under 'u_mass')
        coherenceModel = CoherenceModel(
            model = ldaModel,
            corpus = doc_term_matrix,
            coherence = 'u_mass'
        )
        coherence_score = coherenceModel.get_coherence()
        print("topic num: %d, coherence score: %f" % (num_topics, coherence_score))

        # topics computed by lda
        topic_list = ldaModel.print_topics(num_topics = num_topics, num_words = 20)
        print(topic_list)

        # record results of current model
        with io.open(lda_path + "lda_topics_" + str(num_topics) + '.txt', 'w', encoding = 'utf-8') as f:
            # coherence score
            print(coherence_score, file = f)
            # topics
            for topic in topic_list:
                print(topic[0], "\n", topic[1], "\n", file = f)

        # save current model
        ldaModel.save(lda_path + "lda_topics_" + str(num_topics) + ".model")

    print("----------- LDA training end -----------")


def lda_load(model_path):
    LDA = gensim.models.ldamulticore.LdaMulticore
    ldaModel = LDA.load(model_path)
    return ldaModel


def lda_visual(model, text_corpus):
    import pyLDAvis.gensim
    # get id2word dictionary and term document frequency
    dictionary, corpus = create_lda_input(text_corpus)
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.show(vis)


if __name__ == '__main__':
    with open(input_path, 'r', encoding = 'utf-8') as f:
        image_list = json.load(f)['images']
        f.close()

    # get corpus
    text_corpus = create_corpus(image_list)

    # train lda models
    lda_train(
        text_corpus = text_corpus,
        num_topics_start = 100,
        num_topics_end = 200
    )

    # load a lda model
    ldaModel = lda_load(lda_path + "lda_topics_60.model")

    lda_visual(model = ldaModel, text_corpus = text_corpus)
