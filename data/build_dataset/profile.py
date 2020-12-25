'''
This script will give a simple profile of established dataset.
'''

import os
import json
from collections import Counter

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
folder_path = os.path.join(base_path, 'final')

file_name = [
    'color_light.json',
    'composition.json',
    'dof_and_focus.json',
    'general_impression.json',
    # 'subject.json',
    # 'use_of_camera.json'
]

total_word_freq = Counter()
total_images = Counter()

def cap_num(image_list):
    total = len(image_list)
    train = val = test = 0
    train_cap = val_cap = test_cap = 0
    
    for img in image_list:

        total_images.update([img['filename']])

        if img['split'] == 'train':
            train = train + 1
            train_cap = train_cap + len(img['sentences'])
        elif img['split'] == 'val':
            val = val + 1
            val_cap = val_cap + len(img['sentences'])
        else:
            test = test + 1
            test_cap = test_cap + len(img['sentences'])

    total_cap = train_cap + val_cap + test_cap
    return total, train, val, test, total_cap, train_cap, val_cap, test_cap


def dict_size(image_list, min_word_freq = 5):
    word_freq = Counter()
    
    # recore frequence of each word
    [word_freq.update(cap['tokens']) for img in image_list for cap in img['sentences']]
    [total_word_freq.update(cap['tokens']) for img in image_list for cap in img['sentences']]

    word_num = len(word_freq)
    words_filter = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    filter_word_num = len(words_filter)

    print('Vocabulary Size:', word_num)
    return word_num, filter_word_num


if __name__ == '__main__':

    for item in file_name:

        path = os.path.join(folder_path, item)

        with open(path, 'r', encoding = 'utf-8') as f:
            image_list = json.load(f)['images']
            f.close()
        
        print('\n----------- ', item, ' -----------')
        
        total, train, val, test, total_cap, train_cap, val_cap, test_cap = cap_num(image_list)
        print('Number of images: Total: %d, Train: %d, Val: %d, Test: %d' % (total, train, val, test))
        print('Number of captions: Total: %d, Train: %d, Val: %d, Test: %d' % (total_cap, train_cap, val_cap, test_cap))

        dict_size(image_list, min_word_freq = 0)

        word_cnt = 0
        for img in image_list:
            for cap in img['sentences']:
                word_cnt = word_cnt + len(cap['tokens'])
        
        print('Word count:', word_cnt)
    
    print('\n----------- Total -----------')

    total_word_num = len(total_word_freq)
    print('Vocabulary Size:', total_word_num)

    total_images_num = len(total_images)
    print('Number of images:', total_images_num)