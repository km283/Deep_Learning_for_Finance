import sys, os

# from utils.helpers import CSVParser, HeadlineParser
from helper import CSVParser, HeadlineParser, Reverser, Padder



class Headline:

    def __init__(self, newsfile, trainfile, testfile, valfile, word_index_model):

        self.word_index_model = word_index_model
        self.tokenizer = HeadlineParser()
        with open(newsfile, "r") as newsf:
            self.newsfile = newsf.readlines()

        if (trainfile is not None) and (testfile is not None) and (valfile is not None):
            with open(trainfile, "r") as trainingfile:
                processed_lines = self.process_lines(trainingfile.readlines())
                # self.training_dates = set(trainingfile.readlines())
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
        for line in self.newsfile:
            date, ticker, headline = self.parse_news_line(line, to_vector = True)
            if date in self.training_dates:
                train.append((date, ticker, headline))
            elif date in self.test_dates:
                val.append((date, ticker, headline))
            else:
                test.append((date, ticker, headline))
        return train, val, test

    def minibatch(self, items, batch_size = 10, pad = False):
        if items == None:
            raise ValueError("Item needs to be list.")
        for i in range(len(items) // batch_size):
            start =  i * batch_size
            end = start + batch_size
            batch = items[start: end]
            # sequence_length = list(map(lambda x: len(x[2]), batch))
            sequence_lengths = [len(item[2]) for item in batch]
            max_len = max(sequence_lengths)
            f = lambda x : (x[0], x[1], Reverser.reverse(Padder.padd_list(x[2], 1, max_len, False)))
            batch = list(map(f, batch))
            yield max_len, sequence_lengths, batch

            # lengths = list(map(lambda x: len(x[2]), batch))
            # max_len = max(lengths)
            # print(batch)
            # # print(max_len)
