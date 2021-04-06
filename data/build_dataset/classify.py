"""
Classify captions into 6 aspects based on LDA model trained by `lda.py`:

- color & light (color_light.json)
- subject of image (subject.json)
- composition (composition.json)
- depth of field & focus (dof_and_focus.json)
- general impression (general_impression.json)
- use of camera (use_of_camera.json)

The output files will be saved to `output_path`.
"""

import json
import io
import lda
from tqdm import tqdm

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

input_path = os.path.join(base_path, 'clean.json')
lda_path = os.path.join(base_path, 'lda')
output_path = os.path.join(base_path, 'aspects')


topic_map = {
    '0': 'color_light',
    '1': 'subject',
    '2': 'subject',
    '3': 'composition',
    '4': 'dof_and_focus',
    '5': 'general_impression',
    '6': 'color_light',
    '7': 'subject',
    '8': 'composition',
    '9': 'general_impression',
    '10': 'dof_and_focus',
    '11': 'general_impression',
    '12': 'subject',
    '13': 'color_light',
    '14': 'subject',
    '15': 'subject',
    '16': 'subject',
    '17': 'subject',
    '18': 'subject',
    '19': 'color_light',
    '20': 'subject',
    '21': 'general_impression',
    '22': 'dof_and_focus',
    '23': 'general_impression',
    '24': 'general_impression',
    '25': 'subject',
    '26': 'color_light',
    '27': 'delete',
    '28': 'delete',
    '29': 'general_impression',
    '30': 'delete',
    '31': 'color_light',
    '32': 'general_impression',
    '33': 'composition',
    '34': 'delete',
    '35': 'delete',
    '36': 'color_light',
    '37': 'delete',
    '38': 'delete',
    '39': 'color_light',
    '40': 'delete',
    '41': 'general_impression',
    '42': 'color_light',
    '43': 'use_of_camera',
    '44': 'color_light',
    '45': 'delete',
    '46': 'use_of_camera',
    '47': 'color_light',
    '48': 'dof_and_focus',
    '49': 'delete',
    '50': 'general_impression',
    '51': 'general_impression',
    '52': 'dof_and_focus',
    '53': 'composition',
    '54': 'delete',
    '55': 'delete',
    '56': 'general_impression',
    '57': 'delete',
    '58': 'general_impression',
    '59': 'general_impression',
}


"""
get the topic with max probability

input param:
    topic_list(list[tuple(topicID, topicProb)]): a list of topics with their probability

return:
    (str): name of the final topic
"""
def get_topic(topic_list):

    max_topic_score = 0
    final_topic = -1

    for topic in topic_list:
        if topic[1] > max_topic_score and topic_map[str(topic[0])] != 'delete':
            max_topic_score = topic[1]
            final_topic = topic[0]

    if final_topic == -1:
        return "general_impression"
    else:
        return topic_map[str(final_topic)]


if __name__ == '__main__':
    with open(input_path, 'r', encoding = 'utf-8') as f:
        image_list = json.load(f)['images']
        f.close()

    # get corpus
    text_corpus = lda.create_corpus(image_list)
    # load a lda model
    ldaModel = lda.lda_load(lda_path + "lda_topics_60.model")
    # get dictionary
    dictionary, _ = lda.create_lda_input(text_corpus)

    topic_caps = {
        "color_light": {
            "images": [],
        },
        "general_impression": {
            "images": [],
        },
        "subject": {
            "images": [],
        },
        "dof_and_focus": {
            "images": [],
        },
        "use_of_camera": {
            "images": [],
        },
        "composition": {
            "images": [],
        }
    }

    for cnt, img in enumerate(tqdm(image_list, desc = 'Classified images: ')):
        color_light = {
            "filename": img["filename"],
            "url": img["url"],
            'sentences': []
        }
        general_impression = {
            "filename": img["filename"],
            "url": img["url"],
            'sentences': []
        }
        subject = {
            "filename": img["filename"],
            "url": img["url"],
            'sentences': []
        }
        dof_and_focus = {
            "filename": img["filename"],
            "url": img["url"],
            'sentences': []
        }
        use_of_camera = {
            "filename": img["filename"],
            "url": img["url"],
            'sentences': []
        }
        composition = {
            "filename": img["filename"],
            "url": img["url"],
            'sentences': []
        }

        for cap in img['sentences']:
            corpus = lda.get_all_ngrams(cap)
            bow = dictionary.doc2bow(corpus)
            topic_list = ldaModel.get_document_topics(bow)
            topic = get_topic(topic_list)

            if topic == 'color_light':
                color_light["sentences"].append(cap["clean"])
            elif topic == 'general_impression':
                general_impression["sentences"].append(cap["clean"])
            elif topic == 'subject':
                subject["sentences"].append(cap["clean"])
            elif topic == 'dof_and_focus':
                dof_and_focus["sentences"].append(cap["clean"])
            elif topic == 'use_of_camera':
                use_of_camera["sentences"].append(cap["clean"])
            elif topic == 'composition':
                composition["sentences"].append(cap["clean"])

        if len(color_light["sentences"]) > 0:
            topic_caps["color_light"]["images"].append(color_light)
        if len(general_impression["sentences"]) > 0:
            topic_caps["general_impression"]["images"].append(general_impression)
        if len(subject["sentences"]) > 0:
            topic_caps["subject"]["images"].append(subject)
        if len(dof_and_focus["sentences"]) > 0:
            topic_caps["dof_and_focus"]["images"].append(dof_and_focus)
        if len(use_of_camera["sentences"]) > 0:
            topic_caps["use_of_camera"]["images"].append(use_of_camera)
        if len(composition["sentences"]) > 0:
            topic_caps["composition"]["images"].append(composition)

    # wirte classified captions into json file separately
    with open(output_path + 'color_light.json', 'w') as f:
        json.dump(topic_caps["color_light"], f)
        f.close()
    with open(output_path + 'general_impression.json', 'w') as f:
        json.dump(topic_caps["general_impression"], f)
        f.close()
    with open(output_path + 'subject.json', 'w') as f:
        json.dump(topic_caps["subject"], f)
        f.close()
    with open(output_path + 'dof_and_focus.json', 'w') as f:
        json.dump(topic_caps["dof_and_focus"], f)
        f.close()
    with open(output_path + 'use_of_camera.json', 'w') as f:
        json.dump(topic_caps["use_of_camera"], f)
        f.close()
    with open(output_path + 'composition.json', 'w') as f:
        json.dump(topic_caps["composition"], f)
        f.close()


# if __name__ == '__main__':

#     with open(input_path, 'r', encoding = 'utf-8') as f:
#         image_list = json.load(f)['images']
#         f.close()

#     topic_caps = {
#         "use_of_camera": {
#             "images": [],
#         },
#     }

#     for cnt, img in enumerate(tqdm(image_list, desc = 'Classified images: ')):
#         use_of_camera = {
#             "filename": img["filename"],
#             "url": img["url"],
#             'sentences': []
#         }

#         for cap in img['sentences']:
#             if "exposure" in cap["clean"]\
#                 or "expose" in cap["clean"]\
#                 or "camera" in cap["clean"]\
#                 or "speed" in cap["clean"]\
#                 or "shutter" in cap["clean"]\
#                 or "iso" in cap["clean"]\
#                 or "aperture" in cap["clean"]:
#                 use_of_camera["sentences"].append(cap["clean"])


#         if len(use_of_camera["sentences"]) > 0:
#             topic_caps["use_of_camera"]["images"].append(use_of_camera)


#     with open(output_path + 'use_of_camera.json', 'w') as f:
#         json.dump(topic_caps["use_of_camera"], f)
#         f.close()
