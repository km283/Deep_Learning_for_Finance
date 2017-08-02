import tensorflow as tf
import numpy as np


class Autoencoder:

    def __init__(self, X, y,
                 decoder_inputs,
                 batch_size=100,
                 frame_dim=400,
                 epoch=100,
                 hidden_size=400,
                 sequence_length=None,
                 learning_rate=0.0001,
                 display_step=100,
                 is_training=False,
                 l2_regularization=None,
                 dropout_rate=0.3):

        # Hyperparameters
        # self.num_steps = n_steps
        self.frame_dim = frame_dim
        self.learning_rate = learning_rate
        self.display_step = display_step
        self.hidden_size = hidden_size
        self.is_training = is_training
        self.lambda_l2_reg = l2_regularization
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length

        # inputs must be in the format (batch, size, n_steps frame_dim)
        self.encoder_inputs = tf.contrib.layers.batch_norm(
            X, is_training=self.is_training)
        # self.decoder_inputs = tf.contrib.layers.batch_norm(decoder_inputs,
                                                           # is_training=self.is_training)

        self.decoder_inputs = decoder_inputs
        self.targets = y

        # Decoder needs to unfold the inputs and then add the GO token
        # print("Y shape is", y.shape)
        # dec_values = tf.unstack(tf.transpose(y, perm= [1,0,2]))
        # decoder_inputs = (
        #     [tf.zeros_like(dec_values[0], name="GO")] + dec_values[:-1])
        # self.decoder_inputs = tf.stack(decoder_inputs)

    def initialize_rnn(self, rnn_type="LSTM"):
        with tf.variable_scope("rnn_encoder",
                               initializer=tf.contrib.layers.variance_scaling_initializer(seed=2)):
            if rnn_type == "GRU":
                fw_cell = tf.contrib.rnn.GRUCell(
                    self.hidden_size, activation=tf.nn.elu)
                bw_cell = tf.contrib.rnn.GRUCell(
                    self.hidden_size, activation=tf.nn.elu)
            else:
                fw_cell = tf.contrib.rnn.LSTMCell(
                    self.hidden_size, activation=tf.nn.elu)
                bw_cell = tf.contrib.rnn.LSTMCell(
                    self.hidden_size, activation=tf.nn.elu)

            outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                     self.encoder_inputs,
                                                                     sequence_length=self.sequence_length,
                                                                     dtype=tf.float32)

        with tf.variable_scope("rnn_decoder",
                               initializer=tf.contrib.layers.variance_scaling_initializer(seed=2)):
            decode_data_norm = tf.contrib.layers.batch_norm(
                self.decoder_inputs, is_training=self.is_training)
            # Add dropout
            if rnn_type == "GRU":
                dec_fw_cell = tf.contrib.rnn.GRUCell(
                    self.hidden_size, activation=tf.nn.elu)
                dec_bw_cell = tf.contrib.rnn.GRUCell(
                    self.hidden_size, activation=tf.nn.elu)
            else:
                dec_fw_cell = tf.contrib.rnn.LSTMCell(
                    self.hidden_size, activation=tf.nn.elu)
                dec_bw_cell = tf.contrib.rnn.LSTMCell(
                    self.hidden_size, activation=tf.nn.elu)

            decoder_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                decode_data_norm,
                initial_state_fw=encoder_state[0],
                initial_state_bw=encoder_state[1],
                sequence_length=self.sequence_length,
                time_major=False,
                dtype=tf.float32)

            # BIDirectional --> Batch Norm --> Droput --> Dense(frame_dim)
            decoder_outputs = tf.contrib.layers.batch_norm(
                tf.concat(decoder_outputs, 2), is_training=self.is_training)
            # decoder_outputs = tf.contrib.layers.batch_norm(decoder_outputs[1], is_training =  self.is_training)
            decoder_outputs = tf.contrib.layers.dropout(
                decoder_outputs, self.dropout_rate)
            decoder_outputs = tf.layers.dense(decoder_outputs, self.frame_dim)
        return encoder_state, decoder_outputs

    def loss(self, decoder_outputs, loss_func="Adam", clip_threshold=1.0):
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
