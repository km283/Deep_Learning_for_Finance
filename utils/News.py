import os, sys
import re
import nltk

from utils.helper import CSVParser, HeadlineParser


class News:

    def __init__(self,
                    filename,
                    word_index_model=None,
                    trainfile = None,
                    valfile = None,
                    testfile=None):
        """ This just parses the news
            filename to parse
        """
        self.filename = filename
        self.tokenizer = HeadlineParser()
        self.index = 0
        self.last_batch_index = 0
        self.word_index_model = word_index_model
        with open(filename) as nf:
            self.newsfilelines = nf.readlines()

        if (trainfile is not None) and (testfile is not None) and (valfile is not None):
            with open(trainfile, "r") as trainingfile:
                processed_lines = self.process_lines(trainingfile.readlines())
                self.training_dates = set(processed_lines)
            with open(valfile, "r") as validationfile:
                processed_lines = self.process_lines(validationfile.readlines())
                self.validation_dates = set(processed_lines)
            with open(testfile, "r") as testingfile:
                processed_lines = self.process_lines(testingfile.readlines())
                self.test_dates = set(processed_lines)
        else:
            raise NotImplementedError

        self.train, self.val, self.test = self.initialize()

    def process_lines(self, lines):
        return list(map(lambda x : x.strip(), lines))

    def parse_news_line(self, line, to_vector = False):
        items = CSVParser.parse(line)
        date = items[2]
        ticker = items[0]
        headline = self.tokenizer.parse(items[3])
        # print(headline)
        if to_vector:
            unk = self.word_index_model.word_indexes.get("<UNK>")
            headline = list(map(lambda x: self.word_index_model.word_indexes.get(x, unk), headline))
        return date, ticker, headline

    def initialize(self):
        train = []
        val = []
        test = []
        for line in self.newsfilelines:
            date, ticker, headline = self.parse_news_line(line, to_vector = True)
            if date in self.training_dates:
                train.append((date, ticker, headline))
            elif date in self.test_dates:
                val.append((date, ticker, headline))
            else:
                test.append((date, ticker, headline))
        return train, val, test


    def __len__(self):
        return len(self.newsfilelines)

    def parse_news_line(self, line, to_vector = False):
        items = CSVParser.parse(line)
        date = items[2]
        ticker = items[0]
        headline = self.tokenizer.parse(items[3])
        # print(headline)
        if to_vector:
            unk = self.word_index_model.word_indexes.get("<UNK>")
            headline = list(map(lambda x: self.word_index_model.word_indexes.get(x, unk), headline))
        return date, ticker, headline

    def minibatch(self, item, batch_size):
        for i in range(0, len(item) // batch_size):
            start = i * batch_size
            end = start + batch_size
            batch = item[start: end]
            # batch = map(self.parse_news_line, batch)
            yield batch


    def get_news(self):
        """Gets the news using iterators"""
        # if not (self.index <= len(self.newsfilelines) - 1):
        #     self.index = 0
        if self.index >= self.__len__():
            return None

        line = self.newsfilelines[self.index]
        items = CSVParser.parse(line)
        date = items[2]
        ticker = items[0]
        headine = self.tokenizer.parse(items[3])
        self.index += 1
        return date, ticker, headine
