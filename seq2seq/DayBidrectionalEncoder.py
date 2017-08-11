import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tensorflow as tf
import numpy as np

from Autoencoders.DayBidirectionalEncoderNN import DayBidirectionalEncoderNN
from utils.helper import CSVParser, Padder
from utils.DayHeadlineModel import DayHeadlineModel


# DATADIR = "/cs/home/un4/Documents/Dissertation/Data/"
# # NEWS_ENCODED = DATADIR + "encoded_headlines_.csv"
# NEWS_ENCODED = DATADIR + "encoded_headlines_mt_train.csv"
# NEWS_ENCODED_VAL = DATADIR + "encoded_headlines_mt_val.csv"
# NEWS_ENCODED_TEST = DATADIR + "encoded_headlines_mt_test.csv"
# DAY_ENCODED = DATADIR + "encoded_day_mt_test.csv"
#
#
# SUMDIR = "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/"
# SUMMARYDIR = SUMDIR + "summaries/day_mt_train2/"
# CHECKPOINT = SUMDIR + "checkpoints/day_mt_train2/model.ckpt"


DATADIR = "/cs/home/un4/Documents/Dissertation/Data/"
# NEWS_ENCODED = DATADIR + "encoded_headlines_.csv"
# NEWS_ENCODED = DATADIR + "encoded_headlines_least_squares_train.csv"
# NEWS_ENCODED_VAL = DATADIR + "encoded_headlines_least_squares_val.csv"
# NEWS_ENCODED_TEST = DATADIR + "encoded_headlines_least_squares_test.csv"
# DAY_ENCODED = DATADIR + "encoded_day_ls_train.csv"

NEWS_ENCODED = DATADIR + "encoded_headlines_least_squares_vanilla_train.csv"
NEWS_ENCODED_VAL = DATADIR + "encoded_headlines_least_squares_vanilla_val.csv"
NEWS_ENCODED_TEST = DATADIR + "encoded_headlines_least_squares_vanilla_test.csv"
DAY_ENCODED = DATADIR + "encoded_day_ls_vanilla_test.csv"

SUMDIR = "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/"
SUMMARYDIR = SUMDIR + "summaries/day_ls_train/"
CHECKPOINT = SUMDIR + "checkpoints/day_ls_train/model.ckpt"


def main():
    # filename = "./data/encoded_headlines.csv"
    # checkpoint = "./daybiencoder/model.ckpt"
    day_headline_model = DayHeadlineModel(NEWS_ENCODED)
    day_headline_model_val = DayHeadlineModel(NEWS_ENCODED_VAL)
    day_headline_model_test = DayHeadlineModel(NEWS_ENCODED_TEST)
    # Declearing hyper parameters.
    print("Initializing hyper parameters")
    n_steps = 12
    # batch_size = 32
    batch_size = 32
    frame_dim = 150 #400 # 300 for machine translation.
    display_step = 100
    hidden_size = frame_dim / 2
    learning_rate = 0.0001
    epoch = 200

    device = "/gpu:0"
    with tf.device(device):
        # Inputs.
        print("Initializing inputs")
        s_inputs = tf.placeholder(
            tf.float32, [None, None, frame_dim], name="s_inputs")
        s_outputs = tf.placeholder(
            tf.float32, [None, None, frame_dim], name="s_outputs")

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
                           hidden_size=hidden_size,
                           learning_rate=learning_rate,
                           display_step=display_step,
                           l2_reg=l2_reg,
                           is_training=is_training)
        # Initialize the RNN network
        # TODO: Add Bidirectionality.
        encoder_state, decoder = model.initialize_rnn()

        # Initialize Loss functions: (Adam, RMSProp).
        loss, optimizer = model.loss(decoder, "Adam")

    tf.summary.scalar("loss", loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # prev_val_loss = 999999
    prev_val_loss = 999999

    def process_inputs(items):
        inputs = []
        outputs = []

        for (a, b) in items:
            inputs.append(a)
            outputs.append(b)

        seq_length = np.array(list(map(len, inputs)))
        max_item = len(max(inputs, key=len))

        f = lambda x : Padder.padd_list(x, frame_dim, limit=max_item)
        inputs = list(map(f, inputs))
        outputs = list(map(f, outputs))

        inputs = np.stack(inputs, axis=0)
        outputs = np.stack(outputs, axis=0)

        return inputs, outputs , seq_length

    predicted = True
    # losses = []

    with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                         allow_soft_placement = True)) as sess:
        # saver.restore(sess, checkpoint)
        if not predicted:
            tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(SUMMARYDIR, sess.graph)
            saver.restore(sess, CHECKPOINT)
            # sess.run(init)
            counter = 0
            for ep in range(model.epoch):
                losses = []
                for i, items in enumerate(day_headline_model.minibatch(batch_size, pad=False)):
                    inputs, outputs , seq_length = process_inputs(items)

                    feed_dict = {s_inputs: inputs,
                                 s_outputs: outputs,
                                 is_training: True,
                                 source_sequence_length : seq_length,
                                 target_sequence_length : seq_length,
                                 l2_reg: 0.1}
                    summary, cost = sess.run(
                        [optimizer, loss], feed_dict=feed_dict)
                    losses.append(cost)
                    if counter % display_step == 0:
                        step = ep * batch_size * i
                        summary_writer.add_summary(summary, step)

                        val_losses = []
                        for valitem in day_headline_model_val.minibatch(batch_size, pad=False):
                            val_inputs, val_outputs , seq_length = process_inputs(valitem)
                            feed_dict = {s_inputs: val_inputs,
                                         s_outputs: val_outputs,
                                         is_training: False,
                                         source_sequence_length : seq_length,
                                         target_sequence_length : seq_length,
                                         l2_reg: 0}
                            cost = sess.run([loss], feed_dict=feed_dict)
                            val_losses.append(cost[0])
                        # train_mse = sum(losses) / float(len(losses))
                        val_mse = sum(val_losses)/float(len(val_losses))
                        print("Cost is {}, Val MSE {}, Ep: {}/{}".format(cost[0], val_mse, ep, model.epoch))
                        if prev_val_loss > val_mse:
                            print("saving")
                            prev_val_loss = val_mse
                            saver.save(sess, CHECKPOINT)
                        # Perform Validation chech here
                    counter += 1
                # print(losses)
                mse = sum(losses) / float(len(losses)) if len(losses) > 0 else 0
                print("Epoch 000{}, MSE Loss: {}".format(ep, mse))
        else:
            saver.restore(sess, CHECKPOINT)
            with open(DAY_ENCODED, "w") as encoded_day_file:
                for index, items in enumerate(day_headline_model_test.minibatch(1, full_information=True, pad=False, dimension=frame_dim)):
                    ticker, date, vectors = items[0]
                    inputs = vectors[0]
                    outputs = vectors[1]

                    seq_length = [len(inputs)]
                    inputs = np.stack(inputs, axis=0)
                    inputs = np.expand_dims(inputs, axis = 0)
                    # print(seq_length)
                    feed_dict = {s_inputs: inputs,
                                 is_training: False,
                                 source_sequence_length : seq_length,
                                 l2_reg: 0}

                    enc_states = sess.run([encoder_state], feed_dict = feed_dict)
                    # print(enco)
                    enc_states = enc_states[0][0].h[0]
                    # sys.exit(1)
                    encoded_input = " ".join(list(map(str, enc_states)))
                    encoded_output = "{},{},{}\n".format(ticker, date, encoded_input)
                    encoded_day_file.write(encoded_output)
            encoded_day_file.close()
            print("Finish Saving", DAY_ENCODED)

    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     for epoch in range(self.epoch):
    # for index, items in enumerate(day_headline_model.minibatch(1,
    # pad=True)):
    # summary_writer.close()


if __name__ == "__main__":
    main()
