import tensorflow as tf
import numpy as np


class StackedAutoencoder:

    def __init__(self, X, y,
                 num_layers = 1,
                 batch_size=100,
                 frame_dim=400,
                 epoch=100,
                 hidden_size=400,
                 sequence_length=None,
                 learning_rate=0.0001,
                 display_step=100,
                 is_training=None,
                 regularization=None,
                 dropout_rate=0.3):

        # Hyperparameters
        # self.num_steps = n_steps
        self.num_layers = num_layers
        self.frame_dim = frame_dim
        self.learning_rate = learning_rate
        self.display_step = display_step
        self.hidden_size = hidden_size
        self.is_training = is_training
        self.lambda_l2_reg = regularization
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length

        # inputs must be in the format (batch, size, n_steps frame_dim)
        # self.encoder_inputs = tf.nn.l2_normalize(X, tf.shape(X))
        # self.encoder_inputs = tf.nn.batch_normalization(X, 1)
        # self.encoder_inputs = tf.contrib.layers.batch_norm(
                # X, is_training=self.is_training)

        self.encoder_inputs = X
        self.targets = y

        # Decoder inputs.
        go_token = tf.zeros([batch_size, 1, frame_dim],  dtype= tf.float32)
        ending = tf.strided_slice(self.targets, [0, 0, 0], [batch_size, -1, frame_dim], strides =  [1, 1, 1])
        self.decoder_inputs = tf.concat([go_token, ending], axis = 1)

    def make_multi_cell(self, how_many, activation = None):
        cells = [tf.contrib.rnn.LSTMCell(self.hidden_size) for _ in range(how_many)]
        return tf.contrib.rnn.MultiRNNCell(cells)



    def initialize_rnn(self, activation = None):
        with tf.variable_scope("rnn_encoder",
                               initializer=tf.contrib.layers.variance_scaling_initializer(seed=2)):
            encoder_multi_rnn_cell = self.make_multi_cell(self.num_layers,
                                                        activation)
            outputs, encoder_state = tf.nn.dynamic_rnn(encoder_multi_rnn_cell,
                                                        self.encoder_inputs,
                                                        sequence_length = self.sequence_length,
                                                        time_major=False,
                                                        dtype=tf.float32)



        with tf.variable_scope("rnn_decoder",
                               initializer=tf.contrib.layers.variance_scaling_initializer(seed=2)):
            decoder_multi_rnn_cell = self.make_multi_cell(self.num_layers,
                                                        activation)
            decoder_outputs, _ = tf.nn.dynamic_rnn(decoder_multi_rnn_cell,
                                             self.decoder_inputs,
                                             initial_state = encoder_state,
                                             sequence_length = self.sequence_length,
                                             time_major = False,
                                             dtype = tf.float32)
            decoder_outputs = tf.layers.dense(decoder_outputs, self.frame_dim)
        return encoder_state, decoder_outputs

    def loss(self, decoder_outputs, loss_func="Adam", clip_threshold=5.0):
        # Regularization parameter.
        l2 = self.lambda_l2_reg * sum(
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("BatchNorm" in tf_var.name or "bias" in tf_var.name)
        )
        # Compute loss.
        self.loss = tf.reduce_sum(
            tf.square(tf.subtract(self.targets, decoder_outputs)))
        # Add regularization.
        self.loss = tf.add(self.loss, l2)

        if loss_func == "Adam":
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate)
        else:
            optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -clip_threshold, clip_threshold), var)
                      for grad, var in grads_and_vars]
        self.training_op = optimizer.apply_gradients(capped_gvs)


        return self.loss, self.training_op
