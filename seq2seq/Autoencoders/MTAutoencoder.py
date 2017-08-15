import sys, os
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense




class MTNN:

    def __init__(self, X, y,
     embedding_size,
     batch_size = 64,
     vocab_size = None,
     embedded_matrix = None,
     hidden_size = 100,
     vocab_to_int = None,
     int_to_vocab = None,
     learning_rate = 0.001,
     sequence_length = None,
     max_seq_len = None,
     num_layers=1):

        self.input_data = X
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        self.embedded_matrix = tf.Variable(embedded_matrix, dtype=tf.float32, name="embedded_matrix")
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab
        self.max_target_sequence_length = max_seq_len

        # self.encoder_inputs = X
        self.encoder_inputs = tf.nn.embedding_lookup(self.embedded_matrix, X)

        ending = tf.strided_slice(y, [0, 0], [self.batch_size, -1], [1, 1])
        dec_input = tf.concat(
            [tf.fill([self.batch_size, 1], self.vocab_to_int['<GO>']), ending], 1)
        self.decoder_inputs = tf.nn.embedding_lookup(self.embedded_matrix, dec_input)

        self.targets = y


    def get_model_inputs():
        # pass
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        #
        target_sequence_length = tf.placeholder(
            tf.int32, (None,), name='target_sequence_length')
        max_target_sequence_length = tf.reduce_max(
            target_sequence_length, name='max_target_len')
        source_sequence_length = tf.placeholder(
            tf.int32, (None,), name='source_sequence_length')
        #
        return input_data, targets, lr,\
                    target_sequence_length,\
                    max_target_sequence_length, \
                    source_sequence_length

    def make_cell(self, size, type="LSTM"):
        # cell = tf.contrib.rnn.LSTMCell(size, initializer =
        #             tf.contrib.layers.variance_scaling_initializer(seed=2),
        #             activation = tf.nn.elu)
        cell = tf.contrib.rnn.LSTMCell(size, initializer =
                    tf.contrib.layers.variance_scaling_initializer())

        return cell

    def create_encoder(self):
        with tf.variable_scope("encoder"):
            encoder_embeded_inputs = tf.contrib.layers.embed_sequence(self.input_data,
            self.vocab_size, self.embedding_size)

            encoder_cell = tf.contrib.rnn.MultiRNNCell(
                                [self.make_cell(self.hidden_size) for _ in range(self.num_layers)])

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                    encoder_embeded_inputs,
                                                    sequence_length=self.sequence_length,
                                                    dtype = tf.float32)
        return encoder_outputs, encoder_state

    # def get_decoder_inputs(self, target_data, vocab_to)

    def create_decoder(self, encoder_state):

        cells = [self.make_cell(self.hidden_size) for _ in range(self.num_layers)]
        decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
        output_layer = Dense(self.vocab_size, kernel_initializer=
        tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # You can add source and target sequence length to make this system robust.
        with tf.variable_scope("decode"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.decoder_inputs,
            sequence_length = self.sequence_length, time_major = False)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                training_helper,
                                                                encoder_state,
                                                                output_layer)

            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                impute_finished = True,
                                                                                maximum_iterations = self.max_target_sequence_length)

        with tf.variable_scope("decode", reuse=True):
            start_tokens = tf.tile(tf.constant([self.vocab_to_int['<GO>']], dtype=tf.int32), [
                                   self.batch_size], name='start_tokens')

            # Helper for the inference process.
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedded_matrix,
                                                                        start_tokens,
                                                                        self.vocab_to_int['<EOS>'])

            # Basic decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                inference_helper,
                                                                encoder_state,
                                                                output_layer)

            # Perform dynamic decoding using the decoder
            inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                               impute_finished=True,
                                                                               maximum_iterations=self.max_target_sequence_length)

        return training_decoder_output, inference_decoder_output



    def seq2seq(self):
        _, encoder_state = self.create_encoder()
        training_decoder_output, inference_decoder_output = \
            self.create_decoder(encoder_state)
        return encoder_state, training_decoder_output, inference_decoder_output


    def optimize(self, max_target_lenght):
        encoder_state, training_decoder_output, inference_decoder_output = \
                self.seq2seq()

        training_logits = tf.identity(training_decoder_output.rnn_output, "logits")
        inference_logits = tf.identity(inference_decoder_output.sample_id, name="predictions")
        masks = tf.sequence_mask(self.sequence_length, self.max_target_sequence_length,
                            dtype = tf.float32, name="masks")

        with tf.name_scope("optimization"):
            cost = tf.contrib.seq2seq.sequence_loss(training_logits,
                                                self.targets, masks)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

              # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                                  for grad, var in gradients if grad is not None]
            training_op = optimizer.apply_gradients(capped_gradients)

        return encoder_state, training_op, cost, training_logits, inference_logits
