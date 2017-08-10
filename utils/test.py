import sys, os
import numpy as np
import tensorflow as tf

sys.path.append("/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/s2s/")
sys.path.append("/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/")

from Headline import Headline
from HeadlineEmbeddingModel import HeadlineEmbeddingModel
from WordIndexes import WordsIndexes
from MT import MTNN


KDATADIR = "/cs/home/un4/Documents/Krystof_Folder/"
TRAIN = KDATADIR + "training_dates.csv"
VAL = KDATADIR + "validation_dates.csv"
TEST = KDATADIR + "testing_dates.csv"

DATADIR = "/cs/home/un4/Documents/Dissertation/Data/"
DISTINCT = DATADIR + "hl_distinct.txt"
GLOVE = DATADIR + "glove.840B.300d.txt"
NEWS = DATADIR + "news_reuters.csv"
ENCODED_HEADLINES = DATADIR + "encoded_headlines_mt_val.csv"

SUMDIR = "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/"
SUMMARYDIR = SUMDIR + "summaries/mt_autoencoder/"
CHECKPOINT = SUMDIR + "checkpoints/mt_autoencoder/model.ckpt"

def main():
    print("Creating word indexes")
    word_indexes = WordsIndexes(DISTINCT, GLOVE)
    headline = Headline(NEWS, TRAIN, TEST, VAL, word_indexes)


    print("Creating word embeddings")
    headlines = HeadlineEmbeddingModel(NEWS, processed=False, word_index_model=word_indexes)
    embedded_matrix = headlines.get_embeded_matrix()

    epoch = 100
    learning_rate = 0.001
    hidden_size = 300
    embedding_size = 300
    batch_size = tf.placeholder(tf.int32)

    print("Preparing variables")
    vocab_to_int = word_indexes.word_indexes
    int_to_vocab = word_indexes.indexed_words
    vocab_size = len(vocab_to_int)
    num_layers = 1

    print("Initilising variables")
    with tf.device("/gpu:0"):
        input_data, targets, lr, target_sequence_length, max_target_len,\
                        source_sequence_length = MTNN.get_model_inputs()

        print("Initilising MTNetwork")
        model = MTNN(input_data, targets,
                        embedding_size,
                        batch_size = batch_size,
                        vocab_size = vocab_size,
                        embedded_matrix = embedded_matrix,
                        hidden_size = hidden_size,
                        vocab_to_int = vocab_to_int,
                        int_to_vocab = int_to_vocab,
                        sequence_length =source_sequence_length,
                        learning_rate = lr,
                        num_layers=num_layers)

        # seq2seq = model.optimize(max_target_len)
        encoder_state, training_op, cost, logits, inference_logits = model.optimize(max_target_len)
        tf.summary.scalar("cost", cost)

    def getvalues(items):
        inputs = []
        outputs = []

        for i in items:
            inputs.append(np.array(i[2][0]))
            outputs.append(np.array(i[2][1]))
        try:
            inputs = np.stack(inputs, axis=0)
            outputs = np.stack(outputs, axis=0)
        except:
            print("Error", inputs, outputs)
        return inputs, outputs

    def get_prediction_values(items):
        inputs = []
        outputs = []

        # date = item[0]
        # ticker = item[1]
        dates = []
        tickers = []
        for i in items:
            dates.append(i[0])
            tickers.append(i[1])
            inputs.append(np.array(i[2][0]))
            outputs.append(np.array(i[2][1]))

        inputs = np.stack(inputs, axis=0)
        outputs = np.stack(outputs, axis=0)
        return dates, tickers, inputs, outputs

    mse_losses = 9999
    val_loss_count = 0
    prev_val_loss = 9999
    tf.summary.scalar("val_loss ", prev_val_loss)
    tf.summary.scalar("mse loss", mse_losses)
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
        summary_writer = tf.summary.FileWriter(SUMMARYDIR, sess.graph)
        saver.restore(sess, CHECKPOINT)
        # sess.run(init)
        # for ep in range(epoch):
        #     counter = 0
        #     losses = []
        #     for item in headline.minibatch(headline.train, batch_size):
        #         max_len = item[0]
        #         lenghts = item[1]
        #         values = item[2]
        #
        #         inputs, outputs = getvalues(values)
        #
        #         feed_dict = {
        #             input_data: inputs,
        #             targets: outputs,
        #             lr: learning_rate,
        #             source_sequence_length: lenghts
        #         }
        #         summary, loss, summ= sess.run([training_op, cost, merged], feed_dict = feed_dict)
        #         if counter % 100 == 0:
        #             val_mse = []
        #             for valset in headline.minibatch(headline.val, batch_size):
        #                 val_lengths = valset[1]
        #                 val_inputs, val_targets = getvalues(valset[2])
        #                 val_feed_dict = {
        #                             input_data: val_inputs,
        #                             targets: val_targets,
        #                             lr: learning_rate,
        #                             source_sequence_length: val_lengths
        #                 }
        #                 validation_loss = sess.run([cost], feed_dict = val_feed_dict)
        #                 val_mse.append(validation_loss[0])
        #             validation_mse = sum(val_mse) / float(len(val_mse))
        #             if prev_val_loss is None:
        #                 prev_val_loss = validation_mse
        #             else:
        #                 if prev_val_loss < validation_mse:
        #                     val_loss_count += 1
        #                 else:
        #                     print("Saving :) VAL MSE {}".format(validation_mse))
        #                     saver.save(sess, CHECKPOINT)
        #                     prev_val_loss = validation_mse
        #                     val_loss_count = 0
        #
        #             if val_loss_count >= 30 and ep > 5:
        #                 print("Early Stopping")
        #                 sys.exit(1)
        #
        #             summary_writer.add_summary(summary, counter * ep)
        #             # summary_writer.add_summary(validation_mse, counter * ep)
        #             summary_writer.add_summary(summ, counter * ep)
        #             print("Ep: {}, Cost is {}, Val Loss is {}".format(ep, loss, validation_mse))
        #             losses.append(loss)
        #         counter += 1
        #     print("Epoch 00{}, Cost {}".format(ep, loss))
        #
        #     new_mse_loss = sum(losses) / float(len(losses))
        #     if mse_losses > new_mse_loss:
        #         mse_losses = new_mse_loss
        #
        # sys.exit(1)
        # Prediction
        batch_size_param = 1
        with open(ENCODED_HEADLINES, "w") as encoded_headlines:
            for item in headline.minibatch(headline.val, batch_size_param):
                max_len = item[0]
                lenghts = item[1]
                values = item[2]
                # print(max_len, lenghts, values)
                date, ticker, inputs, outputs = get_prediction_values(values)
                feed_dict = {
                    input_data: inputs,
                    targets: outputs,
                    batch_size: 1,
                    lr: learning_rate,
                    source_sequence_length: lenghts
                }
                # encoded_state = sess.run([encoder_state], feed_dict: feed_dict)
                encoder_inputs = sess.run([encoder_state], feed_dict = feed_dict)

                    # print(encoder_inputs)
                    # print(len(encoder_inputs))
                    # print(len(encoder_inputs[0]))
                    # first_encoded_inputs = encoder_inputs[0][0].h
                top_encoded_inputs = encoder_inputs[0][0].h[0].tolist()

                # top_encoded_inputs = encoder_inputs[0][1].h[0].tolist()
                # f = lambda x : " ".join(list() x[0][1].h[0].tolist())
                # print(len(encoder_inputs))
                # print(len(top_encoded_inputs))
                # print(top_encoded_inputs)
                # sys.exit(1)
                    # print(type(top_encoded_inputs))
                    # print(top_encoded_inputs)
                top_encoded_inputs = " ".join(list(map(str,top_encoded_inputs)))
                formatted_encoded_string = "{},{},{}\n".format(date[0], ticker[0], top_encoded_inputs)
                encoded_headlines.write(formatted_encoded_string)
                # break
        print("Finished writing")










if __name__ == "__main__":
    main()
