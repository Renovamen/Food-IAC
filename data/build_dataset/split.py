'''
Split dataset (of 6 aspects) into train, val and test set and generate tokens for them.
Also split the whole dataset (regardless of the aspects), for training baseline.
'''

import os
import json
import numpy as np
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

input_path = '/Users/zou/Renovamen/Developing/food-iac/data/aspects/'
output_path = '/Users/zou/Renovamen/Developing/food-iac/data/final/'

aspect_name = [
    'color_light.json',
    'composition.json',
    'dof_and_focus.json',
    'general_impression.json',
    'subject.json',
    'use_of_camera.json'
]

# the whole dataset
all_path = '/Users/zou/Renovamen/Developing/food-iac/data/clean.json'

tokenizer = RegexpTokenizer(r'\w+\S*\w*')


'''
split indices of images into train, val and test set (6:1:1)

input param:
    img_num(int): total number of images

return:
    train_indice(list): indices in train set (shuffled)
    val_indice(list): indices in val set (shuffled)
    test_indice(list): indices in test set (shuffled)
'''
def split_indice(img_num):
    # calc number of images in each set
    train_num = int(img_num * 6 / 8)
    val_num = test_num = int(img_num * 1 / 8)
    
    # split indices
    split_indice = np.arange(img_num)
    np.random.shuffle(split_indice) # shuffle all indices
    test_indice = split_indice[0 : test_num]
    val_indice = split_indice[test_num : test_num + val_num]
    train_indice = split_indice[test_num + val_num:]

    return train_indice, val_indice, test_indice


if __name__ == '__main__':

    json_path = [os.path.join(input_path, topic_path) for topic_path in aspect_name]
    json_path.append(all_path)

    for aspectID, path in enumerate(json_path):

        with open(path, 'r', encoding = 'utf-8') as f:
            image_list = json.load(f)['images']
            f.close()

        # get total number of images
        img_num = len(image_list)
        # get splited indice list
        train_indice, val_indice, test_indice = split_indice(img_num)

        split_image_list = []
        for imgID, img in enumerate(tqdm(image_list, desc = 'Splited images: ')):
            if imgID in train_indice:
                img['split'] = 'train'
            elif imgID in val_indice:
                img['split'] = 'val'
            else:
                img['split'] = 'test'
            
            split_cap_list = []
            for cap in img['sentences']:
                new_cap = {} # current caption with it tokens
                if aspectID == len(json_path) - 1:
                    # we are dealing with the whole dataset now
                    new_cap['raw'] = cap['clean']
                    new_cap['tokens'] = tokenizer.tokenize(cap['clean'])
                else:
                    new_cap['raw'] = cap
                    new_cap['tokens'] = tokenizer.tokenize(cap)
                split_cap_list.append(new_cap)
            
            img['sentences'] = split_cap_list
            split_image_list.append(img)
            
        split_data = {}
        split_data['images'] = split_image_list

        # save splited data
        if aspectID == len(json_path) - 1:
            # we are dealing with the whole dataset now
            saved_path = os.path.join(output_path, 'all.json')
        else:
            saved_path = os.path.join(output_path, aspect_name[aspectID])
        with open(saved_path, 'w') as f:
            json.dump(split_data, f)
            f.close()