import os, sys
import tensorflow as tf
import numpy as np
import pandas as pd

sys.path.append(
    "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/")

from tensorflow.contrib.tensorboard.plugins import projector
from numpy import genfromtxt
from utils.helper import CSVParser


# VFILE = "/cs/home/un4/Documents/Dissertation/Data/mthenc_val.csv"
# VFILE = "/cs/home/un4/Documents/Dissertation/Data/mthenc_test.csv"
# VFILE = "/cs/home/un4/Documents/Dissertation/Data/NewMT/mt_henc_val.csv"
VFILE = "/cs/home/un4/Documents/Dissertation/Data/mtd_val.csv"

def main(news):
    # vfile = genfromtxt(VFILE, delimiter=",")
    items = []
    path =os.path.join("vis/emb", "metadata.tsv")
    print(path)

    with open(VFILE, "r") as vf:
        meta_file = open(path, "w")
        meta_file.write("Ticker\tDate\tHeadline\n")
        for item in vf:
            line = item.split(",")
            ticker = line[1]
            date = line[0]
            headlines = news.get(ticker).get(date, ["None"])
            headline = ", ".join(headlines)

            vectors = line[2].split()
            items.append(vectors)
            meta_file.write("{}\t{}\t{}\n".format(line[0], line[1], headline))
        meta_file.close()
    # df = pd.read_table(VFILE, sep=" ", engine="python", header = None)
    myarr = np.array(items).astype(np.float32)
    embedding_var = tf.Variable(myarr, name="word_embedding")

    with tf.Session() as sess:
        sess.run(embedding_var.initializer)
        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name


        embedding.metadata_path = "metadata.tsv"

        summary_writer = tf.summary.FileWriter("vis/emb/")
        projector.visualize_embeddings(summary_writer, config)

        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, "vis/emb/emb.ckpt", 1)
    summary_writer.close()

    os.system("tensorboard --logdir=vis/emb/")


def read_news(news):
    with open(news, "r") as news_file:
        date_dict = {}
        for items in news_file:
            values = CSVParser.parse(items)
            date = values[2]
            ticker = values[0]
            headline = values[3]
            date_ticker = date_dict.get(ticker, None)
            if date_ticker == None:
                date_dict[ticker] = {date: []}
            else:
                date_ticker_vector = date_ticker.get(date, None)
                if date_ticker_vector == None:
                    date_dict[ticker][date] = []
            date_dict[ticker][date].append(headline)
    return date_dict

if __name__ == "__main__":
    argv = sys.argv[1:]
    news = "/cs/home/un4/Documents/Dissertation/Data/news_reuters.csv"
    newsdf  = read_news(news)
    # print(list(newsdf.keys()))
    main(newsdf)
