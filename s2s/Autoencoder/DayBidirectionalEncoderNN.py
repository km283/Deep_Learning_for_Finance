import os
import sys

import tensorflow as tf
import numpy as np


class DayBidirectionalEncoderNN:

    def __init__(self, X, y,
        n_steps,
        source_sequence_length,
        target_sequence_length,
        batch_size=100,
        frame_dim=400,
        epoch=100,
        hidden_size=400,
        learning_rate=0.0001,
        display_step=100,
        is_training = None,
        l2_reg = None,
        dropout_rate = 0.3):
        """
        Constructor
        X: [num_steps, batch_size, n_inputs]
        y: [num_steps, batch_size, n_inputs]
        """
        self.learning_rate = learning_rate
        self.number_of_steps = n_steps
        self.source_sequence_length = source_sequence_length
        self.target_sequence_length = target_sequence_length
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.epoch = epoch
        self.hidden_size = hidden_size
        self.display_step = display_step


        self.lambda_l2_reg = l2_reg

        # self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.is_training = is_training
        self.dropout_rate = dropout_rate

        # batch_normalize_inputs = tf.contrib.layers.batch_norm(
        #     X, is_training=self.is_training)

        # self.encoder_inputs = tf.transpose(
        #     batch_normalize_inputs, perm=[1, 0, 2])

        # Unstack encoder inputs.
        # Prefill first column with zeros.
        # Order with permutation.
        # self.decoder_inputs = (
        #     [tf.zeros_like(self.encoder_inputs[0], name="GO")] + tf.unstack(self.encoder_inputs)[:-1])
        # self.decoder_inputs = tf.stack(self.decoder_inputs)
        # self.decoder_inputs = tf.transpose(self.decoder_inputs, perm=[1, 0, 2])
        # self.encoder_inputs = batch_normalize_inputs
        self.encoder_inputs = X
        self.targets = y

        # ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        # self.decoder_inputs = tf.concat(
        #         [tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
        go_token = tf.zeros([batch_size, 1, frame_dim],  dtype= tf.float32)
        ending = tf.strided_slice(self.targets, [0, 0, 0], [batch_size, -1, frame_dim], strides =  [1, 1, 1])
        self.decoder_inputs = tf.concat([go_token, ending], axis = 1)
        # self.decoder_inputs = tf.strided_slice(y)


    def initialize_rnn(self, activation = tf.nn.elu):
        with tf.variable_scope("rnn_encoder", initializer=tf.contrib.layers.variance_scaling_initializer(seed=2)):
            fw_cell = tf.contrib.rnn.LSTMCell(
                self.hidden_size, activation=activation)
            bw_cell = tf.contrib.rnn.LSTMCell(
                self.hidden_size, activation=activation)

            outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                     self.encoder_inputs,
                                                                     sequence_length  = self.source_sequence_length,
                                                                     dtype=tf.float32)

        with tf.variable_scope("rnn_decoder",
            initializer=tf.contrib.layers.variance_scaling_initializer(seed=2)):
            decode_data_norm = tf.contrib.layers.batch_norm(
                self.decoder_inputs, is_training=self.is_training)
            # Add dropout
            dec_fw_cell = tf.contrib.rnn.LSTMCell(
                self.hidden_size, activation=activation)
            dec_bw_cell = tf.contrib.rnn.LSTMCell(
                self.hidden_size, activation=activation)

            decoder_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                dec_fw_cell,
                dec_bw_cell,
                self.decoder_inputs,
                initial_state_fw=encoder_state[0],
                initial_state_bw=encoder_state[1],
                sequence_length = self.target_sequence_length,
                time_major=False,
                dtype=tf.float32)

            # BIDirectional --> Batch Norm --> Droput --> Dense(frame_dim)
            decoder_outputs = tf.contrib.layers.batch_norm(tf.concat(decoder_outputs, 2), is_training=self.is_training)
            # decoder_outputs = tf.contrib.layers.batch_norm(decoder_outputs[1], is_training =  self.is_training)
            decoder_outputs = tf.contrib.layers.dropout(decoder_outputs, self.dropout_rate)
            decoder_outputs = tf.layers.dense(decoder_outputs, self.frame_dim)


            # decoder_outputs = tf.layers.dense(
            #     decoder_outputs[0], self.frame_dim)

        return encoder_state, decoder_outputs

    # def get_loss(self, encoder_inputs, decoder_outputs):
    #     y_true = tf.unstack(encoder_inputs)
    #     y_pred = tf.unstack(decoder_outputs)
    #     return y_pred, y_true

    def loss(self, decoder_outputs, loss_func="Adam"):

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
            self.optimizer = tf.train.AdamOptimizer(
                self.learning_rate)
        else:
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)

        gradients = self.optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                              for grad, var in gradients if grad is not None]
        self.training_op = self.optimizer.apply_gradients(capped_gradients)

        return self.loss, self.training_op
