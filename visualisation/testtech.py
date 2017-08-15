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
# VFILE = "/cs/home/un4/Documents/Dissertation/Data/tech_enc_test.csv"
# VFILE = "/cs/home/un4/Documents/Dissertation/Data/mthenc_val.csv"
# VFILE = "/cs/home/un4/Documents/Dissertation/Data/NewMT/mt_henc_val.csv"

VFILE = "/cs/home/un4/Documents/Dissertation/Data/mtd_val.csv"

def main(news, tech):
    # vfile = genfromtxt(VFILE, delimiter=",")
    items = []
    path =os.path.join("vis/emb", "metadata.tsv")
    print(path)

    with open(VFILE, "r") as vf:
        meta_file = open(path, "w")
        meta_file.write("Ticker\tDate\tHeadline\tY\n")
        for item in vf:
            line = item.split(",")
            # ticker = line[0]
            # date = line[1]
            ticker = line[1]
            date = line[0]
            # print(ticker)

            headlines = news.get(ticker).get(date, ["None"])
            buy = tech.get(ticker, None)

            if buy == None:
                buy = "NA"
            else:
                buy = buy.get(date, "None")
            headline = ", ".join(headlines)

            vectors = line[2].split()

            items.append(vectors)
            meta_file.write("{}\t{}\t{}\t{}\n".format(ticker, date, headline, buy))
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

def option(item):
    if item == "1":
        return "SELL"
    elif item == "2":
        return "Hold"
    elif item == "3":
        return "BUY"
    else:
        return "NA"



def read_tech(tech):
    with open(tech, "r") as tech:
        date_dict = {}
        for items in tech:
            values = CSVParser.parse(items)
            date = values[1]
            ticker = values[0]
            buy = values[5]
            # hold = values[9]
            # sell = values[13]
            date_ticker = date_dict.get(ticker, None)
            if date_ticker == None:
                date_dict[ticker] = {date: ""}
            else:
                date_ticker_vector = date_ticker.get(date, None)
                if date_ticker_vector == None:
                    date_dict[ticker][date] = ""
            date_dict[ticker][date] = option(buy)
    return date_dict

if __name__ == "__main__":
    argv = sys.argv[1:]
    news = "/cs/home/un4/Documents/Dissertation/Data/news_reuters.csv"
    tech = "/cs/home/un4/Documents/Krystof_Folder/r.csv"
    newsdf  = read_news(news)
    techdf = read_tech(tech)
    # print(list(newsdf.keys()))
    main(newsdf, techdf)
