import tensorflow as tf

class Question_GRU(object):
    def __init__(self, hidden_dimension):

        self.hidden_dimension = hidden_dimension

        with tf.variable_scope(name_or_scope = 'gru_cell', reuse = tf.AUTO_REUSE) as scope:
            self.gru_cell = tf.nn.rnn_cell.GRUCell(self.hidden_dimension)

    def question_embed(self, word_embed):
        with tf.variable_scope(name_or_scope='question_embedding', reuse=tf.AUTO_REUSE) as scope:
            output, self.state = tf.nn.dynamic_rnn(self.gru_cell, word_embed, dtype = tf.float32)

        return self.state