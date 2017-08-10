import sys, os.path
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
sys.path.append(
    "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/")

from utils.model import Model
from utils.helper import CSVParser, Padder
from tensorflow.contrib.layers import variance_scaling_initializer


class DayEncoderNN:

    def __init__(self, X, y,
        n_steps,
        batch_size=100,
        frame_dim=400,
        epoch=100,
        hidden_size=400,
        learning_rate=0.0001,
        display_step=100):

        # Declearing hyper parameters.
        self.learning_rate = learning_rate
        self.number_of_steps = n_steps
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.epoch = epoch
        self.hidden_size = hidden_size
        self.display_step = display_step

        # Declaring Encoder Inputs.
        self.encoder_inputs = [tf.reshape(X, [-1, self.frame_dim])]
        outputs = [tf.reshape(y, [-1, self.frame_dim])]
        self.targets = outputs

        # Decoder inputs.
        self.decoder_inputs = (
            [tf.zeros_like(self.encoder_inputs[0], name="GO")] + self.targets[:-1])
        # weights = [tf.ones_like(targets_t, dtype=tf.float32) for targets_t in self.targets]


    def initialize_rnn(self, rnn_type = "LSTM"):
        # with tf.name_scope("rnn"):
        # with tf.variable_scope("rnn_layer", reuse =True):
        with tf.variable_scope("rnn_layer", initializer=variance_scaling_initializer()):
            if rnn_type == "GRU":
                cell = tf.contrib.rnn.GRUCell(self.hidden_size)
            else:
                cell = tf.contrib.rnn.LSTMCell(self.hidden_size)

            output, encoder_state = tf.contrib.rnn.static_rnn(
                cell, self.encoder_inputs, dtype=tf.float32)

            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, self.frame_dim)
            decoder_outputs, dec_state = tf.contrib.legacy_seq2seq.rnn_decoder(
                self.decoder_inputs, encoder_state, cell)

        return encoder_state, decoder_outputs

    def get_loss(self, encoder_inputs, decoder_outputs):
        y_true = [tf.reshape(encoder_input, [-1])
                  for encoder_input in encoder_inputs]
        y_pred = [tf.reshape(dec_output, [-1])
                  for dec_output in decoder_outputs]
        return y_pred, y_true

    def loss(self, decoder_outputs, loss_func="Adam"):
        """ This takes in the loss function."""
        y_pred, y_true = self.get_loss(self.encoder_inputs, decoder_outputs)
        with tf.name_scope("loss"):
            self.loss = 0
            for i in range(len(y_true)):
                self.loss += tf.reduce_sum(
                    tf.square(tf.subtract(y_pred[i], y_true[i])))
            if loss_func == "Adam":
                self.optimizer = tf.train.AdamOptimizer(
                    self.learning_rate).minimize(self.loss)
            else:
                self.optimizer = tf.train.RMSPropOptimizer(
                    self.learning_rate).minimize(self.loss)
        return self.loss, self.optimizer
