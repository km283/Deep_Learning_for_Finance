import sys
import os.path
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
sys.path.append(
    "/Users/udokanwosu/Documents/Tutorials/Dissertation/ComputationalInvesting/")


from utils.helper import *
from DefaultAutoencoder import Autoencoder


DATADIR = "/Users/udokanwosu/Documents/junk/ParseText/datafiles/"
DISTINCT = DATADIR + "hl_distinct.txt"
GLOVE = DATADIR + "glove.840B.300d.txt"
NEWS = DATADIR + "news.csv"


def main():
    word_indexes = WordsIndexes(DISTINCT, glove_file=GLOVE)
    headline_model = HeadlineEmbeddingModel(
        NEWS, processed=False, word_index_model=word_indexes, with_information=True)

    news = News(NEWS)

    # Hyper Parameters
    learning_rate = 0.0001
    training_epochs = 10000
    display_step = 100
    hidden_layer_size = 400
    # batch_size = 100
    batch_size = 1
    frame_dim = 300
    training_epochs = 100

    # Placeholders
    sequence_inputs = tf.placeholder(tf.float32, shape=(None, None, frame_dim))
    decoder_inputs = tf.placeholder(tf.float32, shape=(None, None, frame_dim))
    sequence_outputs = tf.placeholder(
        tf.float32, shape=(None, None, frame_dim))

    sequence_length = tf.placeholder(tf.int32, shape=(None, ))

    l2_reg = tf.placeholder(tf.float32, name="lambda_l2_reg")
    is_training = tf.placeholder(tf.bool, name='is_training')

    # Autoencoder model.
    autoencoder = Autoencoder(sequence_outputs,
                              sequence_outputs,
                              decoder_inputs,
                              frame_dim=frame_dim,
                              batch_size=batch_size,
                              hidden_size=hidden_layer_size,
                              sequence_length=sequence_length,
                              is_training=is_training,
                              l2_regularization=l2_reg)
    encoder_state, decoder_outputs = autoencoder.rnn()
    loss, optimizer = autoencoder.loss(decoder_outputs)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    prev_loss = None

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            for index, items in enumerate(news.minibatch(batch_size)):

                # GetHeadlines --> Get sequence length --> Convert to Vectors -->
                # Train
                headlines = list(map(lambda x: x[2], items))
                # Gets the sequence length and the max item.
                seq_lengths = np.array(list(map(len, headlines)))
                max_pad = max(seq_lengths)
                f = lambda x: headline_model.get_vectors(
                    x, pad_how_much=max_pad)
                vectors = list(map(f, headlines))

                print(vectors[1])

                inputs = []
                dec_inputs = []
                outputs = []

                for index, (a, b) in enumerate(vectors):
                    inputs.append(np.array(a))
                    outputs.append(np.array(b))

                go_token = np.zeros_like(inputs[0])
                dec_inputs = [go_token] + inputs[:-1]

                inputs = np.stack(inputs, axis=0)
                dec_inputs = np.stack(dec_inputs, axis=0)
                outputs = np.stack(outputs, axis=0)

                feed_dict = {
                    sequence_inputs: inputs,
                    decoder_inputs: dec_inputs,
                    sequence_outputs: outputs,
                    is_training: True,
                    sequence_length: seq_lengths,
                    l2_reg: 0.3
                }

                summary, cost = sess.run(
                    [optimizer, loss], feed_dict=feed_dict)

                if (display_step % 0) == (display_step - 1):
                    print("Ep: {}, Cost: {}.".format(epoch, cost))


if __name__ == "__main__":
    main()
