import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tensorflow as tf
import numpy as np

from utils.helper import CSVParser, Padder


class HeadlineModel:

    def __init__(self, filename):
        """
        Constructor, filename.
        """
        self.lines = None
        self.length = None
        with open(filename, "r") as day_headline_file:
            self.lines = day_headline_file.readlines()

    def __len__(self):
        """ This returns the len of the file. """
        return len(self.lines)

    def max_headline_count(self):
        """
        returns the max headline count
        """
        raise NotImplementedError("You need to implement this feature")


class DayHeadlineModel(HeadlineModel):

    def __init__(self, filename):
        """
        Constructor, filename.
        """
        HeadlineModel.__init__(self, filename)
        self.date_dict = self.group_by_day()

    def group_by_day(self):
        date_dict = {}
        parser = CSVParser()
        for items in self.lines:
            values = parser.parse(items)
            date = values[0]
            ticker = values[1]
            vectors = list(map(float, values[2].split()))
            date_ticker = date_dict.get(ticker, None)
            if date_ticker == None:
                day_vectors = {}

                vectors_list = []
                vectors_list.append(np.array(vectors))
                # print(len(vectors))

                day_vectors[date] = vectors_list
                date_dict[ticker] = day_vectors
            else:
                date_ticker_vector = date_ticker.get(date, None)
                if date_ticker_vector == None:
                    vectors_list = []
                    vectors_list.append(np.array(vectors))
                    date_dict[ticker][date] = vectors_list
                else:
                    date_dict[ticker][date].append(np.array(vectors))
        return date_dict

    def __len__(self):
        if self.length == None:
            length = 0
            for k, v in self.date_dict.items():
                length += len(v.values())
            self.length = length
        return self.length

    def get_stats(self):
        counts = []
        # counts = list(map(len, self.date_dict.values()))
        # counts = np.array(counts)
        # print(set(counts))
        # return np.mean(counts), np.median(counts)

    def max_day_headlines(self):
        # for _, values in self.date_dict:
        # return len(max(self.date_dict.values().values(), key=len))
        max_value = 0
        for k, v in self.date_dict.items():
            # print(len(v.values()))
            max_value = max(max_value, len(max(v.values(), key=len)))
        return max_value

    def min_day_headlines(self):
        # for _, values in self.date_dict:
        min_value = 10000
        for k, v in self.date_dict.items():
            min_value = min(min_value, len(min(v.values(), key=len)))
        return min_value

    def minibatch(self, batch_size, pad=True, full_information=False):
        """ Gets minibatch. """
        # item_batchs = [ h for h in dates.values() for dates in
        # self.date_dict.values()]
        item_batches = []
        if full_information:
            for k, v in self.date_dict.items():
                for k1, v1 in v.items():
                    if pad:
                        item = Padder.padd(v1, 12, pad_int=0, dimension=400)
                    item_batches.append((k, k1, item))
        else:
            for k, v in self.date_dict.items():
                for item in v.values():
                    if pad:
                        item = Padder.padd(item, 12, pad_int=0, dimension=400)
                    item_batches.append(item)

        # print("Item batches", len(item_batches))
        # print("Headline per day", len(item_batches[0]))
        for i in range(0, self.__len__() // batch_size):
            start_i = i * batch_size
            batch = item_batches[start_i: start_i + batch_size]
            yield batch


class DayEncoder:

    def __init__(self, X, y, 
        n_steps, 
        batch_size=100, 
        frame_dim=400,
        epoch=100, 
        hidden_size=400,
        learning_rate=0.0001,
        display_step=100,
        is_training = None,
        dropout_rate = 0.3):
        """
        Constructor
        X: [num_steps, batch_size, n_inputs]
        y: [num_steps, batch_size, n_inputs]
        """
        self.learning_rate = learning_rate
        self.number_of_steps = n_steps
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.epoch = epoch
        self.hidden_size = hidden_size
        self.display_step = display_step


        self.lambda_l2_reg = tf.placeholder(tf.float32, name="lambda_l2_reg")

        # self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.is_training = is_training
        self.dropout_rate = dropout_rate

        batch_normalize_inputs = tf.contrib.layers.batch_norm(
            X, is_training=self.is_training)

        self.encoder_inputs = tf.transpose(
            batch_normalize_inputs, perm=[1, 0, 2])

        # Unstack encoder inputs.
        # Prefill first column with zeros.
        # Order with permutation.
        self.decoder_inputs = (
            [tf.zeros_like(self.encoder_inputs[0], name="GO")] + tf.unstack(self.encoder_inputs)[:-1])
        self.decoder_inputs = tf.stack(self.decoder_inputs)
        self.decoder_inputs = tf.transpose(self.decoder_inputs, perm=[1, 0, 2])

    def initialize_rnn(self, rnn_type="GRU"):
        with tf.variable_scope("rnn_encoder", initializer=tf.contrib.layers.variance_scaling_initializer(seed=2)):
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
                dec_fw_cell,
                dec_bw_cell,
                self.decoder_inputs,
                initial_state_fw=encoder_state[0],
                initial_state_bw=encoder_state[1],
                time_major=True,
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
            tf.square(tf.subtract(self.decoder_inputs, decoder_outputs)))
        # Add regularization.
        self.loss = tf.add(self.loss, l2)

        if loss_func == "Adam":
            self.optimizer = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)
        else:
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)
        return self.loss, self.optimizer


def main():
    filename = "./data/encoded_headlines.csv"
    checkpoint = "./daybiencoder/model.ckpt"
    day_headline_model = DayHeadlineModel(filename)

    # Declearing hyper parameters.
    print("Initializing hyper parameters")
    n_steps = 12
    # batch_size = 100
    batch_size = 1
    frame_dim = 400
    display_step = 100

    # Inputs.
    print("Initializing inputs")
    s_inputs = tf.placeholder(
        tf.float32, [n_steps, batch_size, frame_dim], name="s_inputs")
    s_outputs = tf.placeholder(
        tf.float32, [n_steps, batch_size, frame_dim], name="s_outputs")

    is_training = tf.placeholder(tf.bool, name='is_training')


    # Creating the day encoder Model.
    print("Creating model")
    model = DayEncoder(s_inputs, s_outputs, n_steps,
                       batch_size=batch_size,
                       frame_dim=frame_dim,
                       epoch=100,
                       hidden_size=500,
                       learning_rate=0.0001,
                       display_step=100, 
                       is_training=is_training)
    # Initialize the RNN network
    # TODO: Add Bidirectionality.
    encoder_state, decoder = model.initialize_rnn()

    # Initialize Loss functions: (Adam, RMSProp).
    loss, optimizer = model.loss(decoder, "Adam")

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    prev_loss = None

    with tf.Session() as sess:
        # saver.restore(sess, checkpoint)
        sess.run(init)
        for ep in range(model.epoch):
            for i, items in enumerate(day_headline_model.minibatch(batch_size)):
                inputs = []
                outputs = []

                for (a, b) in items:
                    inputs.append(a)
                    outputs.append(b)

                inputs = np.stack(inputs, axis=0)
                # print(inputs.shape)
                inputs_T = np.transpose(inputs, axes=[1, 0, 2])

                outputs = np.stack(outputs, axis=0)
                # print(outputs.shape)
                outputs_T = np.transpose(outputs, axes=[1, 0, 2])

                feed_dict = {s_inputs: inputs_T,
                             s_outputs: outputs_T,
                             is_training: True}
                summary, cost = sess.run(
                    [optimizer, loss], feed_dict=feed_dict)
                if i % display_step == 0:
                    print("Cost is {} Ep: {}".format(cost, ep))
            if ep % 3 == 2:
                if prev_loss == None:
                    prev_loss = cost
                else:
                    if prev_loss > cost:
                        saver.save(sess, checkpoint)
                        prev_loss = cost
            print("Epoch 000{}, Cost: {}".format(ep, cost))

        # Predictons.
        # TODO: Store the encoded values as (ticker, date, values)
        # with open("./data/encoded_day.csv", "w") as encoded_day_file:
        #     for index, items in enumerate(day_headline_model.minibatch(batch_size, full_information=True)):
        #         inputs = []
        #         ticker, date, vectors = items[0]
        #         for x in vectors[0]:
        #             inputs.append(np.array(x))
        #         inputs = np.stack(inputs, axis = 0)
        #         inputs = np.expand_dims(inputs, 0)
        #         inputs = np.transpose(inputs, axes=[1,0,2])
        #         # break
        #         feed = {s_inputs: inputs}
        #         enc_states = sess.run(encoder_state, feed)
        #         # print(enco)
        #         encoded_input = " ".join(list(map(str, enc_states[-1])))
        #         encoded_output = "{},{},{}\n".format(ticker, date, encoded_input)
        #         encoded_day_file.write(encoded_output)
        # encoded_day_file.close()

    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     for epoch in range(self.epoch):
    # for index, items in enumerate(day_headline_model.minibatch(1,
    # pad=True)):


if __name__ == "__main__":
    main()
