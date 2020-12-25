'''
This script is used to:
1. translate captions to lowercase
2. remove digits, punctions and useless chars (like '\r')
3. remove non-english captions
4. niceeeeeeee -> nice
5. tokenize captions, lemmatize each word, extract unigrams and bigrams
6. calc informativeness score (based on tf-idf) after step 5 and filter captions
7. remove captions with manually selected bad words (like 'congratulations')

The output is 'clean_path'
'''

import json
import re
import os
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from string import digits
from tqdm import tqdm

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

raw_path = os.path.join(base_path, 'raw.json')
clean_path = os.path.join(base_path, 'clean.json')

tokenizer = RegexpTokenizer(r'\w+\S*\w*')
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))
# bad_word_list = [
#     'challenge', 'challenges', 'dpc', 'dpchallenge', 
#     'congrats', 'congratulations', 'congratulation', 
#     'award', 'awards', 'ribbon', 'ribbons',
#     'title', 'titles',
#     'score', 'scores','scored', 'rating', 
#     'comment', 'comments', 'commented', 'critique',
#     'favorites', 'favorite', 'fav',
#     'thanks', 'thank', 
#     'vote', 'voting', 'votes', 'voters', 'voter', 'voted', 
#     'entry', 'entries', 'luck', 'theme'
# ]
bad_word_list = [
    'congrats', 'congratulations', 'congratulation', 
    'award', 'awards', 'ribbon', 'ribbons', 
    "\\u00" # captions with this char must include non-english words
]
replace_word_map = {
    'colour': 'color'
}

subjectivity_threshold = 150
objectivity_threshold = 15

unigram_dict = {} # unigram frenquency
bigram_dict = {} # bigram frenquency


def basic_clean(cap):
    # lowercase
    low_cap = cap.lower()
    # replace some words, for example: colour -> color
    for word in replace_word_map:
        low_cap = low_cap.replace(word, replace_word_map[word])
    # remove digits
    remove_digits = str.maketrans('', '', digits)
    no_digit = low_cap.translate(remove_digits)
    # remove punctions
    no_punction = re.sub(r"[^\w\d'\s]+", ' ', no_digit)   

    tokens = tokenizer.tokenize(no_punction)  
    return ' '.join(w for w in tokens)#, tokens


def reduce_word_length(cap):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1", cap)


def reduce_cap_length(cap):
    clean_cap = ' '.join(map(reduce_word_length, cap.split()))
    return clean_cap


# get pos tag of each word
def get_pos(cap):
    tokens = re.findall(r"[\w']+|[.,!?;]", cap, re.UNICODE)
    token_pos = pos_tag(tokens)
    return token_pos


# lemmatize each word
def lemmatize(token_pos):
    global lemmatizer
    if token_pos[1].startswith('N'):
        return (lemmatizer.lemmatize(token_pos[0], wordnet.NOUN), token_pos[1])
    elif token_pos[1].startswith('V'):
        return (lemmatizer.lemmatize(token_pos[0], wordnet.VERB), token_pos[1])
    elif token_pos[1].startswith('J'):
        return (lemmatizer.lemmatize(token_pos[0], wordnet.ADJ), token_pos[1])        
    elif token_pos[1].startswith('R'):
        return (lemmatizer.lemmatize(token_pos[0], wordnet.ADV), token_pos[1])
    else:
        return token_pos


# check if the caption contains manually selected bad words
def check_bad_words(token_pos_list):
    for token in token_pos_list:
        if token[0] in bad_word_list:
            return False
    return True


# update frenquency of unigram: nouns
def update_unigram_freq(unigram):
    global unigram_dict
    pos = unigram[1]
    if pos in ['NN', 'NNS']:
        try:
           unigram_dict[unigram[0]] += 1
        except KeyError as e:
            unigram_dict[unigram[0]] = 1
        return True
    else:
        return False


# update frenquency of bigram: descriptor-object
# first word: noun, adj, adv
# second word: noun, adj
def update_bigram_freq(bigram):
    global bigram_dict
    if bigram[0][1] in ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'] and bigram[1][1] in ['NN', 'NNS', 'JJ', 'JJR', 'JJS']:
        bigram_word = bigram[0][0] + '_' + bigram[1][0]
        try:
           bigram_dict[bigram_word] += 1
        except KeyError as e:
            bigram_dict[bigram_word] = 1
        return True        
    else:
        return False


def update_ngram_freq(token_pos):
    # get unigrams (except stopwords) and bigrams
    unigrams =  [(i, j) for i, j in token_pos if i not in (stop)]
    bigrams = ngrams(unigrams, 2)
    # filter unigrams and bigrams
    filtered_unigrams = filter(update_unigram_freq, unigrams)
    filtered_bigrams = filter(update_bigram_freq, bigrams)

    return list(filtered_unigrams), list(filtered_bigrams)


# calc informativeness score (based on tf-idf)
def tf_idf(cap):

    global subjectivity_threshold, objectivity_threshold
    global unigram_dict, bigram_dict

    unigram_score = 1.0
    bigram_score = 1.0
    too_objective = True
    too_subjective = True

    for unigram in cap['unigrams']:
        unigram_score *= unigram_score_list[unigram[0]]
    for bigram in cap['bigrams']:
        bigram_score *= bigram_score_list[bigram[0][0] + '_' + bigram[1][0]]
    
    info_score = -np.log(unigram_score * bigram_score)/2

    print(info_score, cap['clean'])
    if info_score <= subjectivity_threshold:
        too_subjective = False
    if info_score >= objectivity_threshold and len(cap['tokens']) >= 5:
        too_objective = False

    final_flag = not (too_subjective or too_objective)

    # if(final_flag == False):
    #     print(info_score, " ", cap['clean'])

    # print("{:0.1e}".format(bigram_score),
    #         "{:0.1e}".format(unigram_score),
    #         "{:0.1f}".format(info_score), 
    #         cap['clean'])

    return final_flag


if __name__ == '__main__':

    # read raw data from file 'raw_path'
    with open(raw_path, 'r', encoding = 'utf-8') as f:
        raw_data = json.load(f)
        f.close()

    # record num of images and captions before cleaning
    raw_imgs = len(raw_data)
    raw_caps = np.sum([len(raw_data[imgID]["comments"]) for imgID in raw_data])
    
    image_list = []

    # clean
    for cnt, imgID in enumerate(tqdm(raw_data, desc = 'Basic clean: ')):
        img = {}
        img["sentences"] = []

        for cap in raw_data[imgID]["comments"]:
            
            # lowercase, remove digits and punctions
            basic_clean_cap = basic_clean(cap)
            # for example: niceeeeeee -> nice
            reduced_cap = reduce_cap_length(basic_clean_cap)

            # tokenized caption and get pos of each token
            token_pos_list = get_pos(reduced_cap)
            # lemmatization
            lemmatized_list = list(map(lemmatize, token_pos_list))

            # ignore captions with manually selected bad words
            if check_bad_words(lemmatized_list) == False:
                continue

            # analys word composition
            unigram_list, bigram_list = update_ngram_freq(lemmatized_list)

            # ignore meaningless captions
            if len(unigram_list) == 0 or len(bigram_list) == 0:
                continue
            
            sentence = {}
            # sentence['raw'] = cap # raw caption
            sentence['clean'] = reduced_cap # cleaned caption
            sentence['tokens'] = tokenizer.tokenize(reduced_cap) # tokens

            sentence['unigrams'] = unigram_list
            sentence['bigrams'] = bigram_list

            img["sentences"].append(sentence)
        

        img["filename"] = imgID # filename of image
        img["url"] = raw_data[imgID]["image_url"] # download url of image

        image_list.append(img)

    # print most common unigrams and bigrams with their frenquency
    # print ('\n'.join([i + '\t'+ str(j) for i, j in Counter(unigram_dict).most_common()]))
    # print ('\n'.join([i + '\t' + str(j) for i, j in Counter(bigram_dict).most_common()]))

    unigram_freq_sum = float(np.sum(list(unigram_dict.values())))
    bigram_freq_sum = float(np.sum(list(bigram_dict.values())))

    # calculate score for each n-gram
    unigram_score_list = dict(zip(unigram_dict.keys(), np.array(list(unigram_dict.values())) / unigram_freq_sum))
    bigram_score_list = dict(zip(bigram_dict.keys(), np.array(list(bigram_dict.values())) / bigram_freq_sum))

    # print most common unigrams and bigrams with their score
    # print ('\n'.join([i + '\t'+ "{:0.1e}".format(j) for i, j in Counter(unigram_score_list).most_common()]))
    # print ('\n'.join([i + '\t' + "{:0.1e}".format(j) for i, j in Counter(bigram_score_list).most_common()]))

    clean_data = {}
    clean_data["images"] = []

    # informativeness filter
    for cnt, img in enumerate(tqdm(image_list, desc = 'Info filter: ')):
        captions = img["sentences"]
        filtered_caps = filter(tf_idf, captions)
        img["sentences"] = list(filtered_caps)
        # remove images with no captions
        if(len(img["sentences"]) > 0):
            clean_data["images"].append(img)
    

    # record num of images and captions after cleaning
    clean_imgs = len(clean_data["images"])
    clean_caps = np.sum([len(img["sentences"]) for img in clean_data["images"]])

    print("raw images: %d, raw captions: %d" % (raw_imgs, raw_caps))
    print("clean images: %d, clean captions: %d" % (clean_imgs, clean_caps))
    print("removed images: %d, %0.2f%%" % (raw_imgs - clean_imgs, (1 - clean_imgs/float(raw_imgs))*100))
    print("removed captions: %d, %0.2f%%" % (raw_caps - clean_caps, (1 - clean_caps/float(raw_caps))*100))


    # write clean data into file 'clean_path'
    with open(clean_path, 'w') as f:
        json.dump(clean_data, f)
        f.close()