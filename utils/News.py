import os, sys
import re
import nltk

from utils.helper import CSVParser, HeadlineParser


class News:

    def __init__(self, filename, word_index_model=None):
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

    def __len__(self):
        return len(self.newsfilelines)

    def get_all_news(self):
        """ Gets the news."""
        all_news = []
        for items in self.newsfilelines:
            items = CSVParser.parse(line)
            date = items[2]
            ticker = items[0]
            headine = self.tokenizer.parse(items[3])
            self.index += 1
            # return date, ticker, headine
            all_news.append(date, ticker, headline)
        return all_news

    def parse_news_line(self, line, to_vector = False):
        items = CSVParser.parse(line)
        date = items[2]
        ticker = items[0]
        headline = self.tokenizer.parse(items[3])
        print(headline)
        if to_vector:
            unk = self.word_index_model.word_indexes.get("<UNK>")
            headline = list(map(lambda x: self.word_index_model.word_indexes.get(x, unk), headline))
        return date, ticker, headline

    def minibatch(self, batch_size):
        for i in range(0, self.__len__() // batch_size):
            start_i = i * batch_size
            batch = map(self.parse_news_line, self.newsfilelines[
                        start_i: start_i + batch_size])
            yield batch

    def auto_encoder_minibatch(self, batch_size):
        """
        Auto encoder minibatch.
        """
        for i in range(0, self.__len__() // batch_size):
            start_i = i * batch_size
            f = lambda x : self.parse_news_line(x, to_vector = True)
            news_batch = self.newsfilelines[start_i: start_i + batch_size]
            batch = map(f, news_batch)
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
