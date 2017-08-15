import sys, os
import tensorflow as tf
import numpy as np

sys.path.append(
    "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/seq2seq")

from technicalhelper import TechnicalData
from Autoencoders.StackedAutoencoder import StackedAutoencoder
from utils.config import DATADIR, SUMDIR



TECHNICALS =  DATADIR + "technical_list_of_arrays.npz"
CHECKPOINT =  SUMDIR + "/checkpoints/tech_enc/"
# SUMMARYDIR =  SUMDIR + "/checkpoints/tc_enc/"
# ENC_TECHNICALS = DATADIR + "tech_enc_test.csv"

def main():

    technical_data = TechnicalData(TECHNICALS)

    predict = True
    hidden_size = 5 * 4
    batch_size = 1 if predict else 128
    learning_rate = 0.01
    l2 = 0.1
    n_steps = 30
    frame_dim = 5
    epoch = 100 * 10
    seq_len = [n_steps for i in range(batch_size)]
    num_layers = 1


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
    saver = tf.train.Saver(max_to_keep=4)
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT)
    previous_loss = 999999
    batch_counter = 0
    early_stop = 0
    with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                         allow_soft_placement = True)) as sess:
        if not predict:
                sess.run(init)
                counter = 0
                for ep in range(epoch):
                    for labels, values in technical_data.minibatch(technical_data.train, batch_size):
                        outputs = values.astype(np.float32)
                        inputs = values[:,::-1,: ]
                        feed_dict = {
                            X: inputs,
                            y: outputs,
                            sequence_length: seq_len,
                            is_training: True,
                            regularization: l2
                        }
                        summary, cost = sess.run([training_op, loss], feed_dict = feed_dict)
                        if counter % 100 == 0:
                            val_losses = []
                            for vallabels, valvalues in technical_data.minibatch(technical_data.val, batch_size):
                                valoutputs = valvalues.astype(np.float32)
                                valinputs = values[:,::-1,: ]
                                feed_dict = {
                                    X: valinputs,
                                    y: valoutputs,
                                    sequence_length: seq_len,
                                    is_training: False,
                                    regularization: 0
                                }
                                valcost = sess.run([loss], feed_dict = feed_dict)
                                val_losses.append(valcost[0])
                            valmse = sum(val_losses)/float(len(val_losses))
                            print("Ep {}, Cost is {}, Val MSE {}.".format(ep, cost, valmse))
                            if valmse < previous_loss:
                                print("Saving :)")
                                saver.save(sess, CHECKPOINT, global_step=batch_counter)
                                previous_loss = valmse
                                early_stop = 0
                            else:
                                early_stop += 1
                            if early_stop > 100:
                                print("Early Stop")
                                sys.exit(1)
                            batch_counter += 1
                        counter += 1
        else:
                saver.restore(sess, latest_checkpoint)
                fileitems = {
                "tech_enc_train.csv": technical_data.train,
                 "tech_enc_val.csv":technical_data.val,
                 "tech_enc_test.csv": technical_data.test
                 }

                for k, v in fileitems.items():
                    with open(DATADIR + k, "w") as enc_file:
                        for labels, values in technical_data.minibatch(v, batch_size):
                            label = labels[0]
                            date, ticker = label[0], label[1]
                            inputs = values.astype(np.float32)
                            inputs = inputs[:,::-1,:]
                            feed_dict = {
                                X: inputs,
                                sequence_length: seq_len,
                                is_training: False,
                                regularization: 0
                            }
                            pred_in = sess.run([encoded_state], feed_dict=feed_dict)
                            pred_in = pred_in[0][0].h[0]
                            # print(pred_in)
                            pred_in = " ".join(list(map(str,pred_in)))
                            formatted_encoded_string = "{},{},{}\n".format(date, ticker, pred_in)
                            # print(formatted_encoded_string)
                            enc_file.write(formatted_encoded_string)
                            # sys.exit(1)
                        print("Finished", k)



if __name__ == "__main__":
    main()
