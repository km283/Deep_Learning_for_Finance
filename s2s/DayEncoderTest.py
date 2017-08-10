import sys, os.path
import tensorflow as tf
import numpy as np

# Import Parent Directory
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from Autoencoder.DayEncoderNN import DayEncoderNN
from utils.model import Model
from utils.helper import CSVParser, Padder
from utils.DayHeadlineModel import DayHeadlineModel

from tensorflow.contrib.layers import variance_scaling_initializer



DATADIR = "/cs/home/un4/Documents/Dissertation/Data/"
NEWS_ENCODED = DATADIR + "encoded_headlines.csv"
CHECKPOINT = "Autoencoder/checkpoints/dayencoder_mt/model.ckpt"





def main():
    """
        This method performs all of the operation.
    """
    day_headline_model = DayHeadlineModel(NEWS_ENCODED)
    print(len(day_headline_model.date_dict))
    print(day_headline_model.min_day_headlines())
    print(day_headline_model.max_day_headlines())
    # sys.exit(1)

    # Declearing hyper parameters.
    print(" Initializing hyper parameters")
    n_steps = 12
    batch_size = 1
    frame_dim = 400
    display_step = 100

    # Inputs.
    print("Initializing inputs")
    s_inputs = tf.placeholder(
        tf.float32, [n_steps, batch_size, frame_dim], name="s_inputs")
    s_outputs = tf.placeholder(
        tf.float32, [n_steps, batch_size, frame_dim], name="s_outputs")

    # Creating the day encoder Model.
    print("Creating model")
    model = DayEncoderNN(s_inputs, s_outputs, n_steps,
                       batch_size=batch_size,
                       frame_dim=frame_dim,
                       epoch=100,
                       hidden_size=500,
                       learning_rate=0.0001,
                       display_step=100)
    # Initialize the RNN network
    # TODO: Add Bidirectionality.
    encoder_state, decoder = model.initialize_rnn()

    # Initialize Loss functions: (Adam, RMSProp).
    loss, optimizer = model.loss(decoder, "Adam")

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    prev_loss = None


    with tf.Session() as sess:
        # saver.restore(sess, CHECKPOINT)

        # BATCH ---> INPUT --> TRAIN
        sess.run(init)
        for ep in range(model.epoch):
            for i, items in enumerate(day_headline_model.minibatch(batch_size, pad=False)):
                print(items)
                inputs = []
                outputs = []
                for (a, b) in items:
                    inputs.append(a)
                    outputs.append(b)

                print(inputs)
                inputs = np.stack(inputs, axis=0)
                print(inputs.shape)
                sys.exit(1)
                # print(inputs.shape)
        #         inputs_T = np.transpose(inputs, axes=[1, 0, 2])

        #         outputs = np.stack(outputs, axis=0)
        #         # print(outputs.shape)
        #         outputs_T = np.transpose(outputs, axes=[1, 0, 2])

        #         feed_dict = {s_inputs: inputs_T, s_outputs: outputs_T}
        #         summary, cost = sess.run([optimizer, loss], feed_dict=feed_dict)
        #         if i % display_step == 0:
        #             print("Cost is {} Ep: {}".format(cost, ep))
        #     if ep % 3 == 2:
        #         if prev_loss == None:
        #             prev_loss = cost
        #         else:
        #             if prev_loss > cost:
        #                 saver.save(sess, checkpoint)
        #                 prev_loss = cost
        #     print("Epoch 000{}, Cost: {}".format(ep, cost))

        # Predictons.
        # TODO: Store the encoded values as (ticker, date, values)
        with open("./data/encoded_day.csv", "w") as encoded_day_file:
            for index, items in enumerate(day_headline_model.minibatch(batch_size, full_information=True)):
                inputs = []
                ticker, date, vectors = items[0]
                for x in vectors[0]:
                    inputs.append(np.array(x))
                inputs = np.stack(inputs, axis = 0)
                inputs = np.expand_dims(inputs, 0)

                inputs = np.transpose(inputs, axes=[1,0,2])

                # break
                feed = {s_inputs: inputs}
                enc_states = sess.run(encoder_state, feed)
                # print(enco)
                encoded_input = " ".join(list(map(str, enc_states[-1])))
                encoded_output = "{},{},{}\n".format(ticker, date, encoded_input)
                encoded_day_file.write(encoded_output)
        encoded_day_file.close()





    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     for epoch in range(self.epoch):
    # for index, items in enumerate(day_headline_model.minibatch(1,
    # pad=True)):


if __name__ == "__main__":
    main()
