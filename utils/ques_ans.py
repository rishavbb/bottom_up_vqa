import os
import json
import re
import pickle  #TODO remove this
from utils import images

def preprocess_question(question):
    new_ques = question.lower()
    new_ques = re.sub('[\?\!\,\`\*\.\;\-]', '',new_ques)
    new_ques = re.sub('[\/]', ' / ', new_ques)
    new_ques = re.sub('[\"]', ' " ', new_ques)
    new_ques = re.sub('[\'\s]', ' ', new_ques)
    new_ques = re.sub('[\(]', ' (', new_ques)
    return new_ques

def execute():
    '''{
        "answer_type": "other",
        "img_id": "COCO_train2014_000000458752",
        "label": {
            "net": 1
        },
        "question_id": 458752000,
        "question_type": "what is this",
        "sent": "What is this photo taken looking through?"
    }'''
    ans_json = json.load(open("data/trainval_ans2label.json"))
    trainval_ques_ans_json = []
    train_val_test_no_images = []
    main_trainval_ques_ans = []
    final_trainval_ques_ans = {}
    ques2idx = {}
    question_maker = 0
    for each_file in ['train.json']: #, 'nominival.json']:   Add this after getting number of epochs
        img_ids = []
        f = json.load(open(os.path.join("data", each_file)))
        for each_ques_ans in f:
            img_ids.append(each_ques_ans['img_id'])
        train_val_test_no_images.append(len(list(set(img_ids))))
        trainval_ques_ans_json += f

    #trainval_ques_ans_json = json.load(open("data/train.json")) + json.load(open("data/nominival.json"))
    print("No of Questions that are read: {}".format(len(trainval_ques_ans_json)))
    for each_ques_ans in trainval_ques_ans_json:
        for each_ans in each_ques_ans["label"]:
            if each_ans in ans_json:
                main_trainval_ques_ans.append(each_ques_ans)
                break
    print("No of Questions on which model will be trained upon: {}".format(len(main_trainval_ques_ans)))

    for each_ques_ans in main_trainval_ques_ans:
        each_ques_ans["sent"] = preprocess_question(each_ques_ans["sent"])
        for each_word in each_ques_ans["sent"].split():
            if each_word not in ques2idx:
                ques2idx[each_word] = question_maker
                question_maker+=1

    test_ques_ans = json.load(open('data/minival.json')) + json.load(open(os.path.join("data", "nominival.json")))  #TODO change nominival.json to test.json
    img_ids = []
    for each_ques_ans in test_ques_ans:
        img_ids.append(each_ques_ans['img_id'])

    train_val_test_no_images.append(len(list(set(img_ids))))

    for each_ques_ans in main_trainval_ques_ans:
        if each_ques_ans["question_type"] not in final_trainval_ques_ans:
            final_trainval_ques_ans[each_ques_ans["question_type"]] = []
        final_trainval_ques_ans[each_ques_ans["question_type"]].append(each_ques_ans)
    print("length of categories {}".format(len(final_trainval_ques_ans)))


    lmao_ans = {}  #TODO remove this
    for each_ques_ans in test_ques_ans:
        each_ques_ans["sent"] = preprocess_question(each_ques_ans["sent"])
        lmao_ans[each_ques_ans["question_id"]] = each_ques_ans["label"]  #TODO remove this

    print(train_val_test_no_images)

    with open("data/final_trainval_ques_ans.p", "wb") as encoded_pickle:   #TODO remove this
        pickle.dump(final_trainval_ques_ans, encoded_pickle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("data/test_ques_ans.p", "wb") as encoded_pickle:   #TODO remove this
        pickle.dump(test_ques_ans, encoded_pickle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("data/ques2idx.p", "wb") as encoded_pickle:  #TODO remove this
        pickle.dump(ques2idx, encoded_pickle, protocol = pickle.HIGHEST_PROTOCOL)
    with open("data/lmao_ans.p", "wb") as encoded_pickle:  #TODO remove this
        pickle.dump(lmao_ans, encoded_pickle, protocol = pickle.HIGHEST_PROTOCOL)

    #train_imgid2idx, test_imgid2idx = images.execute(*train_val_test_no_images)

    return final_trainval_ques_ans, test_ques_ans, ques2idx, train_imgid2idx, test_imgid2idx