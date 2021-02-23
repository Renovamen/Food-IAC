import os
import json

def load_wordmap(data_folder: str, data_name: str):
    '''
    Load word2idx map from json.

    args:
        data_folder (str): folder to store dataset output files
        data_name (str): base name of dataset
    '''

    word_map_file = os.path.join(data_folder, 'wordmap_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    return word_map
