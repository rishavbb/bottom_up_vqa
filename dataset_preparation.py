from utils import ques_ans, images
import h5py
import json
import pickle #TODO remove
import os
import numpy as np
import random
from collections import OrderedDict

class Dataset(object):
    def __init__(self, img_feat_path, ques_ans, ques2idx, imgid2idx,
                 batch_size = 32, if_train = True):

        #if if_train:
        #    assert len(filenames)==3
        #    self.labelized_answers = h5py.File(os.path.join(folder, filenames[2]),'r')['labelized_answer']
        #    self.qtype_ques_id_img_id = pickle.load(open(os.path.join(folder, filename), 'rb'))
        #else:
        #    assert len(filenames)==2
        #    self.image_id_question_id = pickle.load(open(os.path.join(folder, filename), 'rb'))

        self.image_features = h5py.File(img_feat_path,'r')['image_features']
        self.ques_ans = ques_ans
        self.imgid2idx = imgid2idx
        #self.embedded_ques = h5py.File(os.path.join(folder, filenames[1]),'r')['embedded_ques']
        self.batch_size = batch_size
        self.ques2idx = ques2idx
        #self.imgid2idx = pickle.load(open(os.path.join(folder, imgid2idx_filename),'rb'))
        #self.ques_id2idx = pickle.load(open(os.path.join(folder, ques_id2idx_filename),'rb'))

        self.if_train = if_train

    def word_embedding_creation(self, ques2idx):
        f = open('data/glove.6B.300d.txt', encoding="utf8")
        stanford_word_embedding = {}
        for embedding in f:
            word = embedding.split()[0]
            word_embed = list(map(float, embedding.split()[1:]))
            stanford_word_embedding[word] = word_embed

        print('Stanford Word Embeddings Loaded')
        word_embeddings = np.zeros((len(ques2idx), 300))
        for word in ques2idx:
            if word in stanford_word_embedding:
                word_embeddings[ques2idx[word]] = stanford_word_embedding[word]

        print("Word Embedding Loaded")

        #with open("data/word_embeddings.p", "wb") as encoded_pickle:
        #    pickle.dump(word_embeddings, encoded_pickle, protocol=pickle.HIGHEST_PROTOCOL)

        return word_embeddings


    def ques_creation(self, question, word_embeddings, ques2idx, img_id, length=14):
        question = question.split()[:length]
        question =  question + ['-1']*(length - len(question))
        question_embed = []
        no_misses = 0
        for word in question:
            if word in ques2idx:
                question_embed.append(word_embeddings[ques2idx[word]])
            else:
                no_misses+=1
        for i in range(no_misses):
            question_embed.append(word_embeddings[ques2idx['-1']])
        return question_embed


    def ans_creation(self, ans_json, all_ans, no_answers):
        each_labelized_answer = np.zeros(no_answers)
        for each_ans, score in all_ans.items():
            each_labelized_answer[ans_json[each_ans]] = score
        return each_labelized_answer



    def create_data_generator(self):
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
        counter = 0
        word_embeddings = self.word_embedding_creation(self.ques2idx)
        if self.if_train:
            ans_json = json.load(open("data/trainval_ans2label.json"))
            #keys = list(self.qtype_ques_id_img_id.keys())
            #shuffle(keys)
            #self.qtype_ques_id_img_id = OrderedDict(zip(keys, self.qtype_ques_id_img_id.values()))

            #random.shuffle(self.ques_ans)

            keys = list(self.ques_ans.keys())
            random.shuffle(keys)
            self.ques_ans = OrderedDict(zip(keys, self.ques_ans.values()))


            batch_img_data = []
            batch_ques_data = []
            batch_ans_data = []
            check = 0 #TODO dlete this

            #for each_ques_ans in self.ques_ans:
            for each_ques_type in self.ques_ans:
                for each_ques_ans in self.ques_ans[each_ques_type]:
                    img_id = each_ques_ans["img_id"]
                    batch_img_data.append(self.image_features[self.imgid2idx[img_id]])
                    batch_ques_data.append(self.ques_creation(each_ques_ans['sent'], word_embeddings, self.ques2idx, img_id))
                    batch_ans_data.append(self.ans_creation(ans_json, each_ques_ans['label'], len(ans_json)))

            #for q_type in self.qtype_ques_id_img_id:
            #    for ques_id,img_id in self.qtype_ques_id_img_id[q_type]:
            #        batch_img_data.append(self.image_features[self.imgid2idx[img_id]])
                    #batch_ques_data.append(self.embedded_ques[self.ques_id2idx[ques_id]])
                    #batch_ans_data.append(self.labelized_answers[self.ques_id2idx[ques_id]])

            # for img_id in self.image_id_question_id:
            #     for ques_id in self.image_id_question_id[img_id]:
            #         batch_img_data.append(self.image_features[self.imgid2idx[img_id]])
            #         batch_ques_data.append(self.embedded_ques[self.ques_id2idx[ques_id]])
            #         batch_ans_data.append(self.labelized_answers[self.ques_id2idx[ques_id]])

                    if counter<self.batch_size-1:
                        counter+=1
                    else:
                        print(check)
                        check+=1
                        counter = 0
                        yield (batch_img_data, batch_ques_data, batch_ans_data)
                        batch_img_data = []
                        batch_ques_data = []
                        batch_ans_data = []
                yield (batch_img_data, batch_ques_data, batch_ans_data)
                batch_img_data = []
                batch_ques_data = []
                batch_ans_data = []

        else:
            batch_img_data = []
            batch_ques_data = []
            batch_ques_id = []
            for each_ques_ans in self.ques_ans:
                img_id = each_ques_ans["img_id"]
                batch_img_data.append(self.image_features[self.imgid2idx[img_id]])
                batch_ques_data.append(self.ques_creation(each_ques_ans['sent'], word_embeddings, self.ques2idx, img_id))
                batch_ques_id.append(each_ques_ans["question_id"])

            #for img_id in self.image_id_question_id:
            #    for ques_id in self.image_id_question_id[img_id]:
            #        batch_img_data.append(self.image_features[self.imgid2idx[img_id]])
            #        batch_ques_data.append(self.embedded_ques[self.ques_id2idx[ques_id]])
            #        batch_ques_id.append(ques_id)
                if counter<self.batch_size-1:
                    counter+=1
                else:
                    counter = 0
                    yield (batch_img_data, batch_ques_data, batch_ques_id)
                    batch_img_data = []
                    batch_ques_data = []
                    batch_ques_id = []
            yield (batch_img_data, batch_ques_data, batch_ques_id)


if __name__ == "__main__":
    trainval_ques_ans, test_ques_ans, ques2idx, train_imgid2idx, test_imgid2idx = ques_ans.execute()

    #train_imgid2idx = pickle.load(open("data/train_imgid2idx.p","rb"))

    #test_imgid2idx = pickle.load(open("data/test_imgid2idx.p","rb"))

    #test_ques_ans = pickle.load(open("data/test_ques_ans.p", "rb")) #TODO delete this

    #print(len(test_ques_ans), len(test_imgid2idx))

    a = [1,2,3,4]
    print(a[0])
    #for each_key in test_imgid2idx.keys():
    #    if each_key in train_imgid2idx:
    #        print("LMAO")
    #for each_ques_ans in test_ques_ans:
    #    if each_ques_ans["img_id"] in train_imgid2idx:
    #        print("LMAO1")
    #    #if each_ques_ans["img_id"] in test_imgid2idx:
    #    #    print("LMAO2")

    #print('COCO_train2014_000000458752' in test_imgid2idx)

    #ques2idx = pickle.load(open("data/ques2idx.p","rb"))

    #trainval_ques_ans = pickle.load(open("data/final_trainval_ques_ans.p","rb"))

    #d_obj = Dataset('data/train_img_features.hdf5', ques2idx, trainval_ques_ans, train_imgid2idx, 32, True)