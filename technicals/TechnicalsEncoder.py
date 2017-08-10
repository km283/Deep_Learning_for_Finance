import sys, os
import tensorflow as tf
import numpy as np

sys.path.append(
    "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/s2s")
# sys.path.append(os.path.abspath(os.path.join(
#     os.path.dirname(__file__), os.path.pardir)))

from technicalhelper import TechnicalData
from Autoencoder.StackedAutoencoder import StackedAutoencoder


DATADIR = "/cs/home/un4/Documents/Dissertation/Data/"
TECHNICALS =  DATADIR + "technical_list_of_arrays.npz"
CHECKPOINT =  ""


def main():

    technical_data = TechnicalData(TECHNICALS)

    hidden_size = 15
    batch_size = 128
    learning_rate = 0.01
    l2 = 0.1
    n_steps = 30
    frame_dim = 5
    epoch = 100 * 10
    # seq_len = [[n_steps] * batch_size]
    seq_len = [n_steps for i in range(batch_size)]
    num_layers = 3


    device = "/gpu:0"
    with tf.device(device):
        X = tf.placeholder(tf.float32, shape=(None, None, frame_dim))
        y = tf.placeholder(tf.float32, shape=(None, None, frame_dim))
        sequence_length = tf.placeholder(tf.int32, shape=(None, ))
        regularization = tf.placeholder(tf.float32, name="l2_regularization")
        is_training = tf.placeholder(tf.bool, name="is_training")

        autoencoder = StackedAutoencoder(X, y,
            num_layers = num_layers,
            batch_size = batch_size,
            frame_dim = frame_dim,
            epoch = epoch,
            hidden_size = hidden_size,
            sequence_length = sequence_length,
            is_training = is_training,
            regularization = regularization
        )

        encoded_state, decoder_outputs  = autoencoder.initialize_rnn()
        loss, training_op = autoencoder.loss(decoder_outputs)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    previous_loss = None

    with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                         allow_soft_placement = True)) as sess:
        sess.run(init)
        counter = 0
        for ep in range(epoch):
            for item in technical_data.minibatch(technical_data.train, batch_size):
                inputs = item.astype(np.float32)
                outputs = inputs
                feed_dict = {
                    X: inputs,
                    y: outputs,
                    sequence_length: seq_len,
                    is_training: True,
                    regularization: l2
                }

                out = sess.run([decoder_outputs], feed_dict=feed_dict)
                summary, cost = sess.run([training_op, loss], feed_dict = feed_dict)
                if counter % 100 == 0:
                    print("Ep {}, Cost is {}.".format(ep, cost))
                counter += 1
            if previous_loss == None:
                previous_loss = cost
            else:
                if previous_loss > cost:
                    previous_loss = cost



if __name__ == "__main__":
    main()
