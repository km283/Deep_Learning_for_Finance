import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tensorflow as tf
import numpy as np

from Autoencoders.DayBidirectionalEncoderNN import DayBidirectionalEncoderNN
from utils.helper import CSVParser, Padder
from utils.DayHeadlineModel import DayHeadlineModel


DATADIR = "/cs/home/un4/Documents/Dissertation/Data/NewMT/"
# NEWS_ENCODED = DATADIR + "encoded_headlines_.csv"
# NEWS_ENCODED = DATADIR + "mthenc_train.csv"
# NEWS_ENCODED_VAL = DATADIR + "mthenc_val.csv"
# NEWS_ENCODED_TEST = DATADIR + "mthenc_test.csv"
# # DAY_ENCODED = DATADIR + "mtday_test.csv"
# DAY_ENCODED = DATADIR + "mtday_val.csv"

NEWS_ENCODED = DATADIR + "mt_henc_train.csv"
NEWS_ENCODED_VAL = DATADIR + "mt_henc_val.csv"
NEWS_ENCODED_TEST = DATADIR + "mt_henc_test.csv"
# DAY_ENCODED = DATADIR + "mtday_test.csv"

DATADIR = "/cs/home/un4/Documents/Dissertation/Data/"
SUMDIR = "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/"

SUMMARYDIR = SUMDIR + "summaries/mtday/"
CHECKPOINT = SUMDIR + "checkpoints/mtday/"

def main():
    day_headline_model = DayHeadlineModel(NEWS_ENCODED)
    day_headline_model_val = DayHeadlineModel(NEWS_ENCODED_VAL)
    day_headline_model_test = DayHeadlineModel(NEWS_ENCODED_TEST)
    # Declearing hyper parameters.
    print("Initializing hyper parameters")
    n_steps = 12
    predicted = True
    batch_size = 1 if predicted else 32
    frame_dim = 500
    display_step = 100
    hidden_size = 600
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
        loss, optimizer = model.loss(decoder, "Adam")

    training_loss = tf.summary.scalar("loss", loss)
    val_loss = tf.summary.scalar("val_loss", loss)
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=2)
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT)
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

    # losses = []
    batch_counter = 0
    earlystopper = 0

    with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                         allow_soft_placement = True)) as sess:
        # saver.restore(sess, checkpoint)
        if not predicted:
            summary_writer = tf.summary.FileWriter(SUMMARYDIR, sess.graph)
            # saver.restore(sess, CHECKPOINT)
            sess.run(init)
            counter = 0
            for ep in range(model.epoch):
                losses = []
                for i, items in enumerate(day_headline_model_val.minibatch(batch_size, pad=False)):
                    inputs, outputs , seq_length = process_inputs(items)

                    feed_dict = {s_inputs: inputs,
                                 s_outputs: outputs,
                                 is_training: True,
                                 source_sequence_length : seq_length,
                                 target_sequence_length : seq_length,
                                 l2_reg: 0.1}
                    summary, cost, tl = sess.run(
                        [optimizer, loss, training_loss], feed_dict=feed_dict)
                    losses.append(cost)

                    if counter % display_step == 0:
                        val_losses = []
                        for valitem in day_headline_model_val.minibatch(batch_size, pad=False):
                            val_inputs, val_outputs , seq_length = process_inputs(valitem)
                            feed_dict = {s_inputs: val_inputs,
                                         s_outputs: val_outputs,
                                         is_training: False,
                                         source_sequence_length : seq_length,
                                         target_sequence_length : seq_length,
                                         l2_reg: 0}
                            vl, v ,m = sess.run([loss, val_loss, merged], feed_dict=feed_dict)
                            val_losses.append(vl)

                        summary_writer.add_summary(summary, batch_counter)
                        summary_writer.add_summary(tl, batch_counter)
                        summary_writer.add_summary(v, batch_counter)
                        # summary_writer.add_summary(m, batch_counter)
                        batch_counter += 1
                        # train_mse = sum(losses) / float(len(losses))
                        val_mse = sum(val_losses)/float(len(val_losses))
                        print("Cost is {}, Val MSE {}, Ep: {}/{}".format(cost, val_mse, ep, model.epoch))
                        if prev_val_loss > val_mse:
                            print("saving")
                            prev_val_loss = val_mse
                            saver.save(sess, CHECKPOINT, global_step=batch_counter)
                            earlystopper = 0
                        else:
                            earlystopper += 1
                        if earlystopper >= 100:
                            sys.exit(1)
                    counter += 1
                # print(losses)
                mse = sum(losses) / float(len(losses)) if len(losses) > 0 else 0
                print("Epoch 000{}, MSE Loss: {}".format(ep, mse))
        else:
            saver.restore(sess, latest_checkpoint)
            fileitems = {
            "mtd_train.csv": day_headline_model,
             "mtd_val.csv": day_headline_model_val,
             "mtd_test.csv": day_headline_model_test
             }
            for k, v in fileitems.items():
                with open(DATADIR + k, "w") as encoded_day_file:
                    for index, items in enumerate(
                                            v.minibatch(batch_size,
                                                    full_information=True,
                                                    pad=False,
                                                    dimension=frame_dim)):

                        ticker, date, vectors = items[0]
                        pred_input = vectors[0]
                        pred_outputs = vectors[1]


                        pred_seq_length = [len(pred_input)]
                        pred_input = np.stack(pred_input, axis=0)
                        pred_input = np.expand_dims(pred_input, axis = 0)
                        feed_dict = {s_inputs: pred_input,
                                     is_training: False,
                                     source_sequence_length : pred_seq_length,
                                     l2_reg: 0}

                        pred_enc_states = sess.run([encoder_state], feed_dict = feed_dict)
                        pred_enc_states = pred_enc_states[0][0].h[0]
                        pred_in = " ".join(list(map(str, pred_enc_states)))
                        pred_out = "{},{},{}\n".format(date, ticker, pred_in)
                        encoded_day_file.write(pred_out)
            print("Finish Saving")


if __name__ == "__main__":
    main()
