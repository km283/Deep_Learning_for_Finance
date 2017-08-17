import sys
import os.path
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
sys.path.append(
    "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/")


from utils.config import KDATADIR, DATADIR, SUMDIR
from utils.helper import *
from utils.WordIndexes import WordsIndexes
from utils.News import News
from utils.HeadlineEmbeddingModel import HeadlineEmbeddingModel
# from DefaultAutoencoder import Autoencoder
from StackedAutoencoder import StackedAutoencoder



DISTINCT = DATADIR + "hl_distinct.txt"
GLOVE = DATADIR + "glove.840B.300d.txt"
NEWS = DATADIR + "news_reuters.csv"

TRAIN = KDATADIR + "training_dates.csv"
VAL = KDATADIR + "validation_dates.csv"
TEST = KDATADIR + "testing_dates.csv"

SUMMARYDIR = SUMDIR + "summaries/lshl/"
CHECKPOINT = SUMDIR + "checkpoints/lshl/"

def main(use_gpu = False):
    device = "/cpu:0"
    if gpu:
        device = "/gpu:0"

    word_indexes = WordsIndexes(DISTINCT, glove_file=GLOVE)
    headline_model = HeadlineEmbeddingModel(
        NEWS, processed=False, word_index_model=word_indexes, with_information=True)

    # print(headline_model.get_days())
    # sys.exit(1)
    news = News(NEWS, word_indexes, TRAIN, VAL, TEST)

    # Hyper Parameters
    learning_rate = 0.0001
    training_epochs = 10000
    display_step = 100
    predicted = True
    # batch_size_param = 32
    batch_size_param = 32 if not predicted else 1
    frame_dim = 300

    # hidden_layer_size = 400
    hidden_layer_size = frame_dim / 2
    training_epochs = 100

    with tf.device(device):
        # Placeholders
        sequence_inputs = tf.placeholder(tf.float32, shape=[None, None, frame_dim])
        sequence_outputs = tf.placeholder(tf.float32, shape=[None, None, frame_dim])
        sequence_length = tf.placeholder(tf.int32, shape=[None, ])

        l2_reg = tf.placeholder(tf.float32, name="lambda_l2_reg")
        is_training = tf.placeholder(tf.bool, name='is_training')

        # Autoencoder model.
        autoencoder = StackedAutoencoder(sequence_inputs,
                                  sequence_outputs,
                                  frame_dim=frame_dim,
                                  batch_size=batch_size_param,
                                  hidden_size=hidden_layer_size,
                                  sequence_length=sequence_length,
                                  is_training=is_training,
                                  regularization=l2_reg)
        encoder_state, decoder_outputs = autoencoder.initialize_rnn()
        loss, optimizer = autoencoder.loss(decoder_outputs)
    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=4)
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT)
    prev_loss = None

    def getvalues(items):
        headlines = list(map(lambda x: x[2], items))
        # Gets the sequence length and the max item.
        seq_vec_len = list(map(len, headlines))
        max_pad = max(seq_vec_len)

        f = lambda x: headline_model.get_vectors(
            x, pad_how_much=max_pad)

        vectors = list(map(f, headlines))
        inputs = []
        outputs = []

        for index, (a, b) in enumerate(vectors):
            inputs.append(np.array(a))
            outputs.append(np.array(b))

        inputs = np.stack(inputs, axis=0)
        outputs = np.stack(outputs, axis=0)

        return seq_vec_len, inputs, outputs

    def get_prediction_values(items):
        inputs = []
        outputs = []
        seq_length = [len(items)]
        # print(items)

        date = items[0]
        ticker = items[1]
        inputs, outputs = headline_model.get_vectors(items[2])
        inputs = np.expand_dims(np.asarray(inputs), axis = 0)
        outputs = np.expand_dims(np.asarray(outputs), axis = 0)
        # print(date, ticker)
        return date, ticker, seq_length, inputs, outputs



    # with tf.Session() as sess:
    mse = 99999
    val_loss_count = 0
    prev_val_loss = 9999
    training_loss = tf.summary.scalar("loss", loss)
    val_loss = tf.summary.scalar("val_loss", loss)
    merged = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                         allow_soft_placement = True)) as sess:

        if not predicted:
            summary_writer = tf.summary.FileWriter(SUMMARYDIR, sess.graph)
            sess.run(init)
            # saver.restore(sess, CHECKPOINT)
            counter = 0
            batch_counter = 0
            for epoch in range(training_epochs):
                for index, items in enumerate(news.minibatch(news.train, batch_size_param)):
                    # GetHeadlines --> Get sequence length --> Convert to Vectors -->
                    # Train.
                    losses = []
                    seq_vec_len, inputs, outputs = getvalues(items)

                    feed_dict = {
                        sequence_inputs: inputs,
                        sequence_outputs: outputs,
                        is_training: True,
                        sequence_length: seq_vec_len,
                        l2_reg: 0.3
                    }
                    summary, cost , merged_summary = sess.run([optimizer, loss, merged], feed_dict=feed_dict)
                    # summary_writer.add_summary(merged_summary, batch_counter)
                    # Checking for validation loss
                    if counter % display_step == 0:
                        val_mse = []
                        for valset in news.minibatch(news.val, batch_size_param):
                            val_seq_vec_len, val_inputs, val_outputs = getvalues(items)
                            val_feed_dict = {
                                        sequence_inputs: val_inputs,
                                        sequence_outputs: val_outputs,
                                        is_training: False,
                                        sequence_length: val_seq_vec_len,
                                        l2_reg: 0.
                            }
                            validation_loss, m, vl = sess.run([val_loss, merged, loss], feed_dict = val_feed_dict)
                            # summary_writer.add_summary(summary, batch_counter)
                            # summary_writer.add_summary(validation_loss, batch_counter)
                            # summary_writer.add_summary(m, batch_counter)
                            val_mse.append(vl)
                            batch_counter += 1
                        # total_mse = sum(losses)/float(len(losses)) if len(losses) > 0 else cost
                        validation_mse = sum(val_mse) / float(len(val_mse))
                        print("Ep: {}, Item {} Cost: {}. Validation MSE {}".format(
                                            epoch, (index * batch_size_param), cost, validation_mse))

                        if prev_val_loss < validation_mse:
                            val_loss_count += 1
                        else:
                            print("Saving Prev {}, Current {}.".format(prev_val_loss, validation_mse))
                            saver.save(sess, CHECKPOINT, global_step=batch_counter)
                            prev_val_loss = validation_mse
                            val_loss_count = 0

                        if val_loss_count >= 30 and epoch > 5:
                            print("Early Stopping")
                            sys.exit(1)
                    losses.append(cost)
                    counter += 1

                total_mse = sum(losses)/float(len(losses))
                print("Epoch {}, Total MSE {}.".format(epoch, mse))
        else:
            print("Shit")
            saver.restore(sess, latest_checkpoint)
            fileitems = {
            "least_sq_train.csv": news.train,
             "least_sq_val.csv": news.val,
             "least_sq_test.csv": news.test
             }

            for k, v in fileitems.items():
                ps = ""
                with open(DATADIR + k, "w") as encoded_headlines:
                    for item in news.minibatch(v, batch_size_param):
                        date, ticker, seq_len, pinputs, poutputs = get_prediction_values(item[0])
                        feed_dict = {
                                        sequence_inputs: pinputs,
                                        is_training: False,
                                        sequence_length: seq_len,
                            }
                        encoder_inputs = sess.run([encoder_state], feed_dict = feed_dict)
                        top_encoded_inputs = encoder_inputs[0][0].h[0].tolist()
                        top_encoded_inputs = " ".join(list(map(str,top_encoded_inputs)))
                        print(top_encoded_inputs == ps)
                        ps = top_encoded_inputs
                        formatted_encoded_string = "{},{},{}\n".format(date, ticker, top_encoded_inputs)
                        encoded_headlines.write(formatted_encoded_string)
                print("Finished Encoding to {}".format(k))


if __name__ == "__main__":
    args = sys.argv[1:]
    gpu = False
    try:
        if args[0] == "gpu":
            gpu = True
        else:
            gpu = False
    except:
        print("No parameters was set defaulting to cpu")

    main(use_gpu = gpu)
