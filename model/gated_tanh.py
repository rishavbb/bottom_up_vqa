import tensorflow as tf


class Gated_tanh(object):

    def __init__(self, hidden_dimension):
        self.dense1 = tf.layers.Dense(hidden_dimension, activation='tanh')
        self.dense2 = tf.layers.Dense(hidden_dimension, activation='sigmoid')

    def hadamard(self, x):
        y_bar = self.dense1(x)
        g = self.dense2(x)
        y = y_bar * g

        return y