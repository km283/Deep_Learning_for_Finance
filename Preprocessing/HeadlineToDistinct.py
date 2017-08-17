import os, sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append("/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/utils")

from pyspark import SparkConf, SparkContext
from helper import *

utilsFile = "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/utils/helper.py"
pyfiles = [utilsFile]


# DATADIR = "../data/"
DATADIR = "/cs/home/un4/Documents/Dissertation/Data/"
DISTINCT = DATADIR + "hl_distinct.txt"
GLOVE = DATADIR + "glove.840B.300d.txt"
NEWS = DATADIR + "news_reuters.csv"

def initialialize_spark(master):
    conf = SparkConf().setMaster(master).setAppName("Preprocessing")
    sc = SparkContext(conf=conf, pyFiles=pyfiles)
    return conf, sc


def process_headline():
    conf, sc = initialialize_spark("local")
    news = sc.textFile(NEWS)
    parser = HeadlineParser()
    news_words = news.map(CSVParser.parse)\
                     .flatMap(lambda x: parser.parse(x[3]))\
                     .distinct()\
                     .collect()

    with open(DISTINCT, "w") as distinct_files:
        for word in news_words:
            distinct_files.write(word + "\n")
    distinct_files.close()

    # print(news_words)


def main():
    # Gets distinct words from the headline
    # process_headline()


if __name__ == "__main__":
    main()
