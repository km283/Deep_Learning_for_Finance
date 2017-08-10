import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tensorflow as tf
import numpy as np

from Autoencoder.DayBidirectionalEncoderNN import DayBidirectionalEncoderNN
from utils.helper import CSVParser, Padder
from utils.DayHeadlineModel import DayHeadlineModel


DATADIR = "/cs/home/un4/Documents/Dissertation/Data/"
NEWS_ENCODED = DATADIR + "encoded_headlines.csv"
CHECKPOINT = "Autoencoder/checkpoints/daybiencoder_mt/model.ckpt"

def main():
    # filename = "./data/encoded_headlines.csv"
    # checkpoint = "./daybiencoder/model.ckpt"
    day_headline_model = DayHeadlineModel(NEWS_ENCODED)
    # Declearing hyper parameters.
    print("Initializing hyper parameters")
    n_steps = 12
    batch_size = 100
    frame_dim = 400
    display_step = 100

    # Inputs.
    print("Initializing inputs")
    s_inputs = tf.placeholder(
        tf.float32, [n_steps, batch_size, frame_dim], name="s_inputs")
    s_outputs = tf.placeholder(
        tf.float32, [n_steps, batch_size, frame_dim], name="s_outputs")

    l2_reg = tf.placeholder(tf.float32, name="lambda_l2_reg")

    is_training = tf.placeholder(tf.bool, name='is_training')
    source_sequence_length = tf.placeholder(tf.int32, shape = (None,))
    target_sequence_length = tf.placeholder(tf.int32, shape = (None,))


    # Creating the day encoder Model.
    print("Creating model")
    model = DayBidirectionalEncoderNN(s_inputs, s_outputs, n_steps,
                       batch_size=batch_size,
                       frame_dim=frame_dim,
                       source_sequence_length=source_sequence_length,
                       target_sequence_length=target_sequence_length,
                       epoch=100,
                       hidden_size=500,
                       learning_rate=0.0001,
                       display_step=100,
                       l2_reg=l2_reg,
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
            for i, items in enumerate(day_headline_model.minibatch(batch_size, pad=False)):
                inputs = []
                outputs = []

                for (a, b) in items:
                    inputs.append(a)
                    outputs.append(b)

                # print(len(inputs))

                seq_length = list(map(len, inputs))
                max_item = len(max(inputs, key=len))

                f = lambda x : Padder.padd_list(x, frame_dim, limit=max_item)
                inputs = list(map(f, inputs))
                outputs = list(map(f, outputs))
                inputs = np.stack(inputs, axis=0)
                inputs_T = np.transpose(inputs, axes=[1, 0, 2])
                # print(inputs_T.shape)

                outputs = np.stack(outputs, axis=0)
                # print(outputs.shape)
                outputs_T = np.transpose(outputs, axes=[1, 0, 2])

                feed_dict = {s_inputs: inputs_T,
                             s_outputs: outputs_T,
                             is_training: True,
                             source_sequence_length : seq_length,
                             target_sequence_length : seq_length,
                             l2_reg: 0.3}
                summary, cost = sess.run(
                    [optimizer, loss], feed_dict=feed_dict)
                if i % display_step == 0:
                    print("Cost is {} Ep: {}".format(cost, ep))
            # if ep % 3 == 2:
            if prev_loss == None:
                prev_loss = cost
                saver.save(sess, CHECKPOINT)
            else:
                if prev_loss > cost:
                    saver.save(sess, CHECKPOINT)
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
