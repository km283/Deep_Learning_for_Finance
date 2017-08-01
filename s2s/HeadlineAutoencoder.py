import sys
import tensorflow as tf
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.helper import *


# DATADIR = "../data/"
DATADIR = "/Users/udokanwosu/Documents/junk/ParseText/datafiles/"
DISTINCT = DATADIR + "hl_distinct.txt"
GLOVE = DATADIR + "glove.840B.300d.txt"
NEWS = DATADIR + "news.csv"

#  This initializes the word indexes using glove and so forth.
word_indexes = WordsIndexes(DISTINCT, glove_file=GLOVE)

# This creates the headline embedding model.
headline_model = HeadlineEmbeddingModel(
    NEWS, processed=False, word_index_model=word_indexes, with_information=True)
# embedded_matrix = headlines.get_embeded_matrix(glove_file=GLOVE)
# print(embedded_matrix[:4])
# print(headlines.indexes[: 3])

news = News(NEWS)
# line = news.get_news()

# Parameters
# learning_rate = 0.0001
# training_epochs = 10000
# display_step = 100

learning_rate = 0.0001
training_epochs = 10000
display_step = 100

# Network Parameters
# the size of the hidden state for the lstm (notice the lstm uses 2x of this amount so actually lstm will have state of size 2)
# size = 250

size = 400
# 2 different sequences total
# Training Batch size
# batch_size = 100

# Prediction batch size
# batch_size = 100
batch_size = 1
# the maximum steps for both sequences is 5
n_steps = 53
# each element/frame of the sequence has dimension of 3
frame_dim = 300




# the sequences, has n steps of maximum size
seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, frame_dim])
seq_output = tf.placeholder(tf.float32, [n_steps, batch_size, frame_dim])
# what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]
# early_stop = tf.placeholder(tf.int32, [batch_size])

# inputs for rnn needs to be a list, each item/frame being a timestep.
# we need to split our input into each timestep, and reshape it because split keeps dims by default
encoder_inputs = [tf.reshape(seq_input, [-1, frame_dim])]
outputs = [tf.reshape(seq_output, [-1, frame_dim])]
# if encoder input is "X, Y, Z", then decoder input is "0, X, Y, Z". Therefore, the decoder size
# and target size equal encoder size plus 1. For simplicity, here I droped the last one.
decoder_inputs = ([tf.zeros_like(encoder_inputs[0], name="GO")] + encoder_inputs[:-1])
# targets = encoder_inputs
targets = outputs
weights = [tf.ones_like(targets_t, dtype=tf.float32) for targets_t in targets]

# basic LSTM seq2seq model
cell = tf.contrib.rnn.GRUCell(size)
_, enc_state = tf.contrib.rnn.static_rnn(cell, encoder_inputs, dtype=tf.float32)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, frame_dim)
dec_outputs, dec_state = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs, enc_state, cell)

# e_stop = np.array([1, 1])

# flatten the prediction and target to compute squared error loss
# y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]

# Define loss and optimizer, minimize the squared error
loss = 0
for i in range(len(y_true)):
    loss += tf.reduce_sum(tf.square(tf.subtract(y_pred[i], y_true[i])))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
checkpoint = "./ckpt2/model.ckpt"
summary_path = "./summary/"
previous_loss = None

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, checkpoint)
    # sess.run(init)
    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    # Training cycle
    # for epoch in range(training_epochs):
    #     # rand = np.random.rand(n_steps, batch_size, frame_dim).astype('float32')
    #     # x = np.arange(n_steps * batch_size * frame_dim)
    #     # x = x.reshape((n_steps, batch_size, frame_dim))
    #     for index, items in enumerate(news.minibatch(batch_size)):
    #         headlines = map(lambda x: x[2], items)
    #         # print(headlines)
    #         # print(list(headlines))
    #         # headlines = map(lambda x: x[2], items)
    #         f = lambda x : headline_model.get_vectors_from_headline(x, pad_len=53)
    #         # vectors = list(map(lambda x: np.array(headline_model.get_vectors_from_headline(x, pad_len=53)), headlines))
    #         vectors = list(map(f, headlines))

    #         inputs = []
    #         outputs = [] 
    #         for (a, b) in vectors:
    #             inputs.append(np.array(a))
    #             outputs.append(np.array(b))

    #         # print(inputs[1])
    #         # print(outputs[1])

    #         inputs = np.stack(inputs, axis=0)
    #         outputs = np.stack(outputs, axis=0)

    #         # sys.exit(1)
    #         # TODO: get inputs and outputs in different lists

    #         # vectors = np.stack(vectors, axis=0)
    #         inputs_T = np.transpose(inputs, axes=[1,0,2])
    #         outputs_T = np.transpose(outputs, axes=[1,0,2])

    #         # print(result.shape)
    #         feed = {seq_input: inputs_T, seq_output: outputs_T}
    #         # Fit training using batch data
    #         summary, cost_value = sess.run([optimizer, loss], feed_dict=feed)
    #          # Display logs per epoch step
    #         if batch_size % display_step == 0:       
    #             a = sess.run(y_pred, feed_dict=feed)
    #             b = sess.run(y_true, feed_dict=feed)
    #             print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost_value))

    #         if batch_size % 100 == 0:
    #             summary_writer.add_summary(summary, epoch)

    #         if epoch % 4 == 3:
    #             summary_writer.add_summary(summary, epoch)
    #             if previous_loss is None:
    #                 previous_loss = cost_value
    #             if cost_value < previous_loss: 
    #                 print("Saveing")
    #                 saver.save(sess, checkpoint)
    #                 previous_loss = cost_value


    # Prediction file
    print("Performing prediction")
    with open("./data/encoded_headlines.csv", "w") as encoded_file:
        # for index, items in enumerate(news.minibatch(1)):
        #     # headline = map(lambda x : x[2], items)
        #     date, ticker, headline = list(items)[0]
        #     # print(headline)
        #     f = lambda x : np.array(headline_model.get_vectors_from_headline(x, pad_len=53))
        #     vectors = list(map(f, [headline]))
        #     vectors = np.stack(vectors, axis=0)
        #     t = np.transpose(vectors, axes=[1,0,2])


        for index, items in enumerate(news.minibatch(batch_size)):
            date, ticker, headline = list(items)[0]
            # print(date, ticker, headline)
            f = lambda x : headline_model.get_vectors_from_headline(x, pad_len=53)
            vectors = list(map(f, [headline]))
            # print(vectors)

            inputs = []
            for (a, _) in vectors:
                inputs.append(np.array(a))

            inputs = np.stack(inputs, axis=0)
            inputs_T = np.transpose(inputs, axes=[1,0,2])
            feed = {seq_input: inputs_T}
            encoded_states= sess.run(enc_state, feed)
            # print(encoded_states[-1])
            # print(encoded_input)
            encoded_input = " ".join(list(map(str, encoded_states[-1])))
            encoded_output = "{},{},{}\n".format(date, ticker, encoded_input)
            encoded_file.write(encoded_output)
    encoded_file.close()


    print("Optimization Finished!")