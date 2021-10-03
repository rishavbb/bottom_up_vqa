import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

import tensorflow as tf

from model.question_model import Question_GRU
from model.top_down_model import Top_Down
from model.gated_tanh import Gated_tanh
from data_preparation import Dataset
import time
import pickle
from tqdm import tqdm
import json


class Model(object):
    def __init__(self, hidden_dimension, no_of_image_regions,
                 total_no_of_answers, batch_size, epochs=20,
                 clip_rate=3.0):

        tf.reset_default_graph()

        self.no_of_image_regions = no_of_image_regions
        self.hidden_dimension = hidden_dimension
        self.gru = Question_GRU(self.hidden_dimension)
        self.top_down = Top_Down(self.no_of_image_regions, self.hidden_dimension)
        self.gated_tanh1 = Gated_tanh(self.hidden_dimension)
        self.gated_tanh2 = Gated_tanh(self.hidden_dimension)
        self.gated_tanh3 = Gated_tanh(self.hidden_dimension)

        self.batch_size = batch_size
        self.total_no_of_answers = total_no_of_answers
        self.is_training = tf.placeholder(tf.bool)

        #         self.dataset_iterator = self.retrieve_dataset()
        #         self.input_data = self.dataset_iterator.get_next()

        self.img = tf.placeholder(tf.float32, (None, self.no_of_image_regions, 2048))
        self.ques = tf.placeholder(tf.float32, (None, 14, 300))

        #         self.w1 = tf.layers.Dense(self.hidden_dimension)
        #         self.w2 = tf.layers.Dense(self.hidden_dimension)
        #         self.w3 = tf.layers.Dense(self.hidden_dimension)
        self.w4 = tf.layers.Dense(self.total_no_of_answers)

        self.batch_ques_indicies = tf.placeholder(tf.float32, shape=[None])
        self.epochs = epochs
        self.clip_rate = clip_rate
        self.ans = tf.placeholder(tf.float32, (None, self.total_no_of_answers))
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.architecture()
        self.loss_optimizer()

    def architecture(self):

        self.img = tf.math.l2_normalize(self.img)

        ques_state = self.gru.question_embed(self.ques)
        img_features = self.top_down.weighted_sum(self.img, ques_state)

        ques_state = self.gated_tanh1.hadamard(ques_state)

        img_features = self.gated_tanh2.hadamard(img_features)

        element_wise_product = ques_state * img_features

        #         h = tf.layers.dropout(tf.layers.batch_normalization(tf.nn.sigmoid(self.w3(element_wise_product)),
        #                                                             training = self.is_training),
        #                               rate = 0.15, training = self.is_training)

        h = self.gated_tanh3.hadamard(element_wise_product)

        self.p_y = self.w4(h)

        self.max_possible_ans = tf.argmax(self.p_y, axis=1)

    def loss_optimizer(self):
        with tf.variable_scope(name_or_scope='op', reuse=tf.AUTO_REUSE) as scope:
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.max_possible_ans,
                                                                                  labels=self.ans))
            #             self.optimize = tf.train.AdadeltaOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

            gradients, variables = zip(*self.optimizer.compute_gradients(self.cost))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_rate)
            self.optimize = self.optimizer.apply_gradients(zip(gradients, variables))

    #             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #             self.optimize = tf.group([optimize, update_ops])

    #         grads_and_vars = self.optimizer.compute_gradients(self.cost)
    #         capped_grads_and_vars = [(tf.clip_by_norm(grad, 0.25), var) for grad, var in grads_and_vars]
    #         opt = self.optimizer.apply_gradients(capped_grads_and_vars, global_step=self.global_step)

    #         self.gradients = tf.gradients(self.cost, tf.trainable_variables())
    #         self.clipped_grads, _ = tf.clip_by_global_norm(self.gradients, self.clip_rate)
    #         self.opt = self.optimizer.apply_gradients(zip(self.clipped_grads, tf.trainable_variables()))

    def retrieve_dataset(self, ques2idx, ques_ans, imgid2idx, if_train="train"):

        if if_train == "train":

            # obj = Dataset(folder='data', filenames = ['train_img_features.hdf5',
            #                                          'final_train_embedded_ques.hdf5',
            #                                          'train_labelized_answer.hdf5'],
            #              imgid2idx_filename = 'train_imgid2idx.p',
            #              ques_id2idx_filename = 'train_ques_id2idx.p',
            #              image_id_question_id_filename = 'train_image_id_question_id.p',
            #              batch_size = self.batch_size , if_train = True)
            obj = Dataset('data/train_img_features.hdf5', ques_ans, ques2idx, imgid2idx, self.batch_size, True)

            dataset = tf.data.Dataset.from_generator(obj.create_data_generator,
                                                     output_types=(tf.float32, tf.float32, tf.float32),
                                                     output_shapes=(
                                                     tf.TensorShape([None, self.no_of_image_regions, 2048]),
                                                     tf.TensorShape([None, 14, 300]),
                                                     tf.TensorShape([None, self.total_no_of_answers])))

        elif if_train == "test":

            # obj = Dataset(folder='data', filenames = ['test_img_features.hdf5',
            #                                          'final_test_embedded_ques.hdf5'],
            #              imgid2idx_filename = 'test_imgid2idx.p',
            #              ques_id2idx_filename = 'test_ques_id2idx.p',
            #              image_id_question_id_filename = 'test_image_id_question_id.p',
            #              batch_size = self.batch_size , if_train = False)

            obj = Dataset('data/test_img_features.hdf5', ques_ans, ques2idx, imgid2idx, self.batch_size, False)

            dataset = tf.data.Dataset.from_generator(obj.create_data_generator,
                                                     output_types=(tf.float32, tf.float32, tf.int32),
                                                     output_shapes=(
                                                     tf.TensorShape([None, self.no_of_image_regions, 2048]),
                                                     tf.TensorShape([None, 14, 300]),
                                                     tf.TensorShape([None])))

        elif if_train == "test_dev":

            obj = Dataset(folder='data', filenames=['test_img_features.hdf5',
                                                    'final_test_dev_embedded_ques.hdf5'],
                          imgid2idx_filename='test_imgid2idx.p',
                          ques_id2idx_filename='test_dev_ques_id2idx.p',
                          image_id_question_id_filename='test_dev_img_id_question_id.p',
                          batch_size=self.batch_size, if_train=False)

            dataset = tf.data.Dataset.from_generator(obj.create_data_generator,
                                                     output_types=(tf.float32, tf.float32, tf.int32),
                                                     output_shapes=(
                                                     tf.TensorShape([None, self.no_of_image_regions, 2048]),
                                                     tf.TensorShape([None, 14, 300]),
                                                     tf.TensorShape([None])))

        iterator = dataset.make_initializable_iterator()

        return iterator

    def test(self, ques2idx, test_ques_ans, test_imgid2idx):

        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            ckpt = tf.train.get_checkpoint_state('checkpoint/')

            if ckpt and ckpt.model_checkpoint_path:
                print("Previously trained model loaded")
                module_file = tf.train.latest_checkpoint('checkpoint/')
                saver.restore(sess, module_file)

            else:
                print("No trained model found")
                return False

            # self.dataset_iterator = self.retrieve_dataset(if_train = "test")
            self.dataset_iterator = self.retrieve_dataset(ques2idx, test_ques_ans, test_imgid2idx, if_train="test")
            self.input_data = self.dataset_iterator.get_next()

            idx2ans = pickle.load(open('data/idx2ans.p', 'rb'))
            sess.run(self.dataset_iterator.initializer)

            vqa_ans = []
            try:
                while True:
                    i, q, q_idx = sess.run(self.input_data)
                    feed_dict = {self.img: i,
                                 self.ques: q,
                                 self.batch_ques_indicies: q_idx,
                                 self.is_training: False}
                    ans = sess.run([self.max_possible_ans], feed_dict=feed_dict)[0][0]

                    for each_ans_id, q_i in zip(ans, q_idx):
                        a = str(idx2ans[each_ans_id])
                        d = {}
                        d['answer'] = a
                        d['question_id'] = q_i
                        vqa_ans.append(d)
            except tf.errors.OutOfRangeError:
                print("Testing Over")

        return vqa_ans

    def predict(self, ques2idx, test_ques_ans, test_imgid2idx, sess):

        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            ckpt = tf.train.get_checkpoint_state('checkpoint/')

            if ckpt and ckpt.model_checkpoint_path:
                print("Previously trained model loaded")
                module_file = tf.train.latest_checkpoint('checkpoint/')
                saver.restore(sess, module_file)

            else:
                print("No trained model found")
                return False

            # self.dataset_iterator = self.retrieve_dataset(if_train = "test")
            self.dataset_iterator = self.retrieve_dataset(ques2idx, test_ques_ans, test_imgid2idx, if_train="test")
            self.input_data = self.dataset_iterator.get_next()

            idx2ans = json.load(open('data/trainval_label2ans.json'))
            lmao_ans = pickle.load(open("data/lmao_ans.p", "rb"))

            sess.run(self.dataset_iterator.initializer)
            print(lmao_ans)

            vqa_ans = []
            score = 0
            lmaoo = 0
            try:
                while True:
                    i, q, q_idx = sess.run(self.input_data)
                    feed_dict = {self.img: i,
                                 self.ques: q,
                                 self.batch_ques_indicies: q_idx,
                                 self.is_training: False}
                    ans = sess.run([self.max_possible_ans], feed_dict=feed_dict)[0][0]
                    print(lmaoo)
                    lmaoo += 1

                    for each_ans_id, q_i in zip(ans, q_idx):
                        a = str(idx2ans[int(each_ans_id)])
                        d = {}
                        d['answer'] = a
                        d['question_id'] = q_i
                        vqa_ans.append(d)

            except tf.errors.OutOfRangeError:
                print("Testing Over")
            for each_ans in vqa_ans:
                ques_id = each_ans['question_id']
                ans = each_ans['answer']
                org_answers = lmao_ans[ques_id]
                if ans in org_answers:
                    score += 1
            print("Total right = {}".format(score))
            print("Percenatge = {}".format(score / len(lmao_ans)))

        return vqa_ans

    def train(self, ques2idx, trainval_ques_ans, train_imgid2idx):

        self.dataset_iterator = self.retrieve_dataset(ques2idx, trainval_ques_ans, train_imgid2idx, if_train="train")
        self.input_data = self.dataset_iterator.get_next()

        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            ckpt = tf.train.get_checkpoint_state('checkpoint/')
            if ckpt and ckpt.model_checkpoint_path:
                print("Using previously trained model")
                module_file = tf.train.latest_checkpoint('checkpoint/')
                saver.restore(sess, module_file)
                if_previous = True
            else:
                sess.run(tf.global_variables_initializer())
                if_previous = False

            sess.run(self.dataset_iterator.initializer)

            #             if if_previous:
            #               lr_var = 1e-3
            #             else:
            lr_var = 0.001
            test_ques_ans = pickle.load(open("data/test_ques_ans.p", "rb"))  # TODO delete this
            test_imgid2idx = pickle.load(open("data/test_imgid2idx.p", "rb"))  # TODO dlete this
            self.test_dataset_iterator = self.retrieve_dataset(ques2idx, test_ques_ans, test_imgid2idx,
                                                               if_train="test")  # TODO remove this
            self.test_input_data = self.test_dataset_iterator.get_next()  # TODO remove this
            sess.run(self.test_dataset_iterator.initializer)  # TODO remove this

            try:
                for epoch in tqdm(range(1, self.epochs + 1)):
                    starting_time = time.time()
                    batch_index = 0
                    total_loss = 0
                    try:
                        while True:
                            i, q, a = sess.run(self.input_data)

                            feed_dict = {self.img: i,
                                         self.ques: q,
                                         self.ans: a,
                                         self.learning_rate: lr_var,
                                         self.is_training: True}
                            c, _, lr = sess.run([self.cost, self.optimize, self.learning_rate], feed_dict=feed_dict)
                            total_loss += c
                            batch_index += 1
                            # if batch_index % 20==0:


                    #                                 print("Loss: %.3f\tTime Taken:%.3f\tLearning rate:%.4f" %(total_loss/batch_index,
                    #                                                                               time.time() - starting_time,
                    # #                                                                              lr))
                    #                             if total_loss/batch_index < claim:# and lr_var>0.001:
                    #                                 lr_var = 0.01
                    #                                 claim -= 0.23
                    # #                                     claim-=0.5
                    #                                     print("claim",str(claim))

                    #                                 total_loss = 0
                    #                                 batch_index = 0
                    #                                 starting_time = time.time()
                    except tf.errors.OutOfRangeError:
                        print("Iterating finished")
                        sess.run(self.dataset_iterator.initializer)
                        print("Loss: %.3f\tTime Taken:%.3f\tLearning rate:%.4f" % (total_loss / batch_index,
                                                                                   time.time() - starting_time,
                                                                                   lr))
                        saver.save(sess, 'checkpoint/model')
                        print("Checkpoint made")

                    #                         if epoch<5 and not if_previous:
                    #                           lr_var /= 10
                    # if epoch<=1:
                    #    lr_var = 0.5
                    ################ EACH TESTING ####################
                    # self.dataset_iterator = self.retrieve_dataset(ques2idx, test_ques_ans, test_imgid2idx, if_train = "test")
                    # self.test_input_data = self.test_dataset_iterator.get_next()

                    idx2ans = json.load(open('data/trainval_label2ans.json'))
                    lmao_ans = pickle.load(open("data/lmao_ans.p", "rb"))

                    vqa_ans = []
                    score = 0
                    lmaoo = 0
                    try:
                        while True:
                            i, q, q_idx = sess.run(self.test_input_data)
                            feed_dict = {self.img: i,
                                         self.ques: q,
                                         self.batch_ques_indicies: q_idx,
                                         self.is_training: False}
                            ans = sess.run([self.max_possible_ans], feed_dict=feed_dict)[0][0]
                            print(lmaoo)
                            lmaoo += 1

                            for each_ans_id, q_i in zip(ans, q_idx):
                                a = str(idx2ans[int(each_ans_id)])
                                d = {}
                                d['answer'] = a
                                d['question_id'] = q_i
                                vqa_ans.append(d)

                    except tf.errors.OutOfRangeError:
                        print("Testing Over")
                    for each_ans in vqa_ans:
                        ques_id = each_ans['question_id']
                        ans = each_ans['answer']
                        org_answers = lmao_ans[ques_id]
                        if ans in org_answers:
                            score += 1
                    print("Total right = {}".format(score))
                    print("Percenatge = {}".format(score / len(lmao_ans)))
                    sess.run(self.test_dataset_iterator.initializer)
                    # self.predict(ques2idx, test_ques_ans, test_imgid2idx)


            except KeyboardInterrupt:
                print('Interrupted by user')


# if __name__=="__main__":
#    ans2idx = pickle.load(open('data/ans2idx.p','rb'))
#    ob = Model(512, 36, len(ans2idx), 512, 40)
#    ob.train()

if __name__ == "__main__":
    trainval_ques_ans, test_ques_ans, ques2idx, train_imgid2idx, test_imgid2idx = ques_ans.execute()
    train_imgid2idx = pickle.load(open("data/train_imgid2idx.p", "rb"))

    test_imgid2idx = pickle.load(open("data/test_imgid2idx.p", "rb"))

    ques2idx = pickle.load(open("data/ques2idx.p", "rb"))

    trainval_ques_ans = pickle.load(open("data/final_trainval_ques_ans.p", "rb"))
    test_ques_ans = pickle.load(open("data/test_ques_ans.p", "rb"))

    ans2idx = json.load(open("data/trainval_ans2label.json"))
    ques2idx['-1'] = len(ques2idx)  # TODO this needs to go to ques.py
    ob = Model(512, 36, len(ans2idx), 512, 40)
    ob.train(ques2idx, trainval_ques_ans, train_imgid2idx)

    # d_obj = Dataset('data/train_img_features.hdf5', ques2idx, trainval_ques_ans, train_imgid2idx, 32, True)