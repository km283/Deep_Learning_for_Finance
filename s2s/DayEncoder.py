import tensorflow as tf
import numpy as np
from helper import CSVParser, Padder


class HeadlineModel:

    def __init__(self, filename):
        """ 
        Constructor, filename.
        """
        self.lines = None
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
        length = 0
        for k, v in self.date_dict.items():
            length += len(v.values())
        return length

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

    def minibatch(self, batch_size, pad=True):
        """ Gets minibatch. """
        # item_batchs = [ h for h in dates.values() for dates in self.date_dict.values()]
        item_batches = []
        for k, v in self.date_dict.items():
            for item in v.values():
                if pad:
                    item = Padder.padd(item, 12, pad_int=0, dimension=400)
                item_batches.append(item)

        print("Item batches", len(item_batches))
        print("Headline per day", len(item_batches[0]))
        for i in range(0, self.__len__() // batch_size):
            start_i = i * batch_size
            batch = item_batches[start_i: start_i + batch_size]
            yield batch


class DayEncoder:

    def __init__(self, X, y, n_steps, batch_size=100, frame_dim=400, epoch=100, hidden_size=400, learning_rate=0.0001, display_step=100):
        self.learning_rate = learning_rate
        self.number_of_steps = n_steps
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.epoch = epoch
        self.hidden_size = hidden_size
        self.display_step = display_step

        # self.training_epochs = tra
        # with tf.name_scope("inputs"):


        self.encoder_inputs = [tf.reshape(X, [-1, self.frame_dim])]
        outputs = [tf.reshape(y, [-1, self.frame_dim])]

        self.decoder_inputs = (
            [tf.zeros_like(self.encoder_inputs[0], name="GO")] + self.encoder_inputs[:-1])
        self.targets = outputs
        # weights = [tf.ones_like(targets_t, dtype=tf.float32) for targets_t in self.targets]


    def initialize_rnn(self, rnn_type = "GRU"):
        # with tf.name_scope("rnn"):
        # with tf.variable_scope("rnn_layer", reuse =True):
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

            # y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
            # y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]
            # return y_pred, y_trye=

    def get_loss(self, encoder_inputs, decoder_outputs):
        y_true = [tf.reshape(encoder_input, [-1])
                  for encoder_input in encoder_inputs]
        y_pred = [tf.reshape(dec_output, [-1])
                  for dec_output in decoder_outputs]
        return y_pred, y_true

    def loss(self, decoder_outputs, loss_func="Adam"):
        """ This takes in the loss function """
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

    # def train(self, X, Y, device = None, restore=True):
    #     init = tf.global_variables_initializer()
    #     with tf.Session() as sess:
    #         if restore:
    #             saver.restore(sess, self.checkpoint_path)
    #         else:
    #             sess.run(init)
    #         for epoch in range(self.epoch):
    #             for index, items in enumerate(news.minibatch(batch_size)):


def main():
    filename = "./data/encoded_headlines.csv"
    checkpoint = "./dayencoder/model.ckpt"
    day_headline_model = DayHeadlineModel(filename)
    n_steps = 12
    batch_size = 100
    frame_dim = 400 
    display_step = 100
    s_inputs = tf.placeholder(
        tf.float32, [n_steps, batch_size, frame_dim], name="s_inputs")
    s_outputs = tf.placeholder(
        tf.float32, [n_steps, batch_size, frame_dim], name="s_outputs")
    model = DayEncoder(s_inputs, s_outputs, n_steps,
                       batch_size=batch_size,
                       frame_dim=frame_dim,
                       epoch=100,
                       hidden_size=500,
                       learning_rate=0.0001,
                       display_step=100)
    _, decoder = model.initialize_rnn()
    loss, optimizer = model.loss(decoder, "Adam")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() 
    prev_loss = None

    with tf.Session() as sess:
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

                feed_dict = {s_inputs: inputs_T, s_outputs: outputs_T}
                summary, cost = sess.run([optimizer, loss], feed_dict=feed_dict)
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


    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     for epoch in range(self.epoch):
    # for index, items in enumerate(day_headline_model.minibatch(1,
    # pad=True)):


if __name__ == "__main__":
    main()
