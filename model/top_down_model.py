import tensorflow as tf
from model.gated_tanh import Gated_tanh

class Top_Down(object):

    def __init__(self, no_of_image_regions, hidden_dimension):

        self.no_of_image_regions = no_of_image_regions

        # self.dense1 = tf.layers.Dense(hidden_dimension, activation = 'tanh')
        # self.dense2 = tf.layers.Dense(hidden_dimension, activation = 'sigmoid')
        self.dense1 = tf.layers.Dense(1, activation = None, use_bias = False)
        self.gated_tanh =  Gated_tanh(hidden_dimension)

    def weighted_sum(self, img, ques):

        temp = tf.expand_dims(ques, 1)
        temp = tf.tile(temp, (1, self.no_of_image_regions, 1))
        temp = tf.concat((img, temp), 2)
        #
        # y_bar = self.dense1(temp)
        # g = self.dense2(temp)
        # y = y_bar * g

        y = self.gated_tanh.hadamard(temp)

        a_i = self.dense1(y)

        alpha = tf.nn.softmax(a_i, 1)

        v_hat = img * alpha

        self.weighted_sum = tf.reduce_mean(v_hat, 1)

        return self.weighted_sum