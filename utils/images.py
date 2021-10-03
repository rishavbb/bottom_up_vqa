import csv
import os
import numpy as np
import base64
import json
import pickle  #TODO remove this
import h5py
csv.field_size_limit(100000000)

TRAIN_IMG_FEAT_PATH = ["data/img_features/train2014_obj36.tsv"]# TODO, "data/img_features/val2014_obj36.tsv"]
TEST_IMG_FEAT_PATH = "data/img_features/val2014_obj36.tsv" # TODO keep right only   "data/img_features/test2015_obj36.tsv"
#FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def execute(train_img_length, test_img_length):

    print("Extracting Image Features.")
    train_img_file = h5py.File('data/train_img_features.hdf5','w')
    #val_img_file = h5py.File('data/val_img_features.hdf5', 'w')
    test_img_file =  h5py.File('data/test_img_features.hdf5', 'w')


    #img_ids = {}

    #for name, file in zip(names,question_files):
    #    questions = json.load(open(os.path.join(folder,file)))['questions']
    #    img_ids[name] = []
    #    for ques in questions:
    #        img_ids[name].append(ques['image_id'])
    #    img_ids[name] = list(set(img_ids[name]))
    #    print(file,"loaded successfully.")


    train_img_features = train_img_file.create_dataset('image_features',
                                                       (train_img_length, 36, 2048),
                                                       dtype = 'f')

    #val_img_features = val_img_file.create_dataset('image_features',
    #                                               (val_img_length, 36, 2048),
    #                                                dtype = 'f')

    test_img_features = test_img_file.create_dataset('image_features',
                                                    (test_img_length,36, 2048),
                                                    dtype = 'f')

    train_imgid2idx = {}
    train_idx2imgid = {}
    #val_imgid2idx = {}
    #val_idx2imgidx = {}
    test_imgid2idx = {}
    test_idx2imgid = {}

    train_imgid_maker = 0
    #val_imgid_maker = 0
    test_imgid_maker = 0

    print('Extracting Image Features Starts')

    for each_img_path in TRAIN_IMG_FEAT_PATH:
        with open(each_img_path, "r") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for idx, item in enumerate(reader):
                num_boxes = int(item['num_boxes'])
                image_id = item['img_id']
                features = np.frombuffer(base64.b64decode(item['features']), dtype = np.float32).reshape((num_boxes, -1))

                train_imgid2idx[image_id] = train_imgid_maker
                train_idx2imgid[train_imgid_maker] = image_id
                train_img_features[train_imgid_maker] = features
                train_imgid_maker+=1

        print("Done with storing {} image features".format(each_img_path))
    print(len(train_imgid2idx))

    with open(TEST_IMG_FEAT_PATH, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for idx, item in enumerate(reader):
            num_boxes = int(item['num_boxes'])
            image_id = item['img_id']
            features = np.frombuffer(base64.b64decode(item['features']), dtype = np.float32).reshape((num_boxes, -1))
            test_imgid2idx[image_id] = test_imgid_maker
            test_idx2imgid[test_imgid_maker] = image_id
            test_img_features[test_imgid_maker] = features
            test_imgid_maker+=1

    print(len(test_imgid2idx))
    print("Done with storing {} image features".format(TEST_IMG_FEAT_PATH))


    with open("data/train_imgid2idx.p", "wb") as encoded_pickle:
        pickle.dump(train_imgid2idx, encoded_pickle, protocol = pickle.HIGHEST_PROTOCOL)
    with open("data/test_imgid2idx.p", "wb") as encoded_pickle:
        pickle.dump(test_imgid2idx, encoded_pickle, protocol = pickle.HIGHEST_PROTOCOL)

    return train_imgid2idx, test_imgid2idx