import os
import sys
import csv
import nltk
import re
import numpy as np

from nltk.tokenize import RegexpTokenizer


class Headline:

    def __init__(self, filename, processed=True, word_index_model=None):
        """
        Reads the news headline and then returns a vector of those values.
        filename: this represents the file name of the file.
        wordmodel: this is an instance of the word class that returns the index2word and word2index.
        """
        self.word_index_model = word_index_model
        if processed:
            self.headline_indexes = self.processed_file(filename)
        else:
            self.headline_indexes = self.not_processed(filename)

    def not_processed(self, filename):
        """ This function run when the file has not been processed
            filename: information filename.
            wordmodel: word model from class Word.
        """
        parser = HeadlineParser()
        with open(filename, "r") as headlines_file:
            indexed_headlines = []
            for line in headlines_file:
                items = CSVParser().parse(line)
                date = items[2]
                ticker = items[0]
                tokenized_headline = parser.parse(items[3])
                indexes = self.convert_line_to_index(
                    tokenized_headline)
                indexed_headlines.append(indexes)
        return indexed_headlines

    def processed_file(self, filename):
        """ This function only executes if the file has been processed
            filename: information filename.
            wordmodel: word model from class Word.
        """
        self.filename = filename
        indexed_headlines = []
        with open(filename, "r") as headlines_file:
            for line in headlines_file:
                _, _, indexes = self.process_line(line)
                if len(indexes) > 0:
                    indexes = list(map(float, indexes))
                    headlines.append(indexes)
        return indexed_headlines

    def convert_line_to_index(self, line):
        """
        Takes in a list of words and convert them to their corresponding index.
        line: list of line of words.
        wordmodel: the word model containint the word2index.
        """
        if not isinstance(line, list):
            print(type(line))
            raise ValueError("line needs to be a list")
        return list(map(lambda x: self.word_index_model.word_indexes.get(x, "<UNK>"), line))

    def get_embeded_matrix(self, glove_file = None):
        glove_size = 300
        word_embedding_dict = self.word_index_model.get_word_glove_vectors(glove_file)
        embedding_matrix = np.random.randn(len(self.word_index_model.word_indexes), glove_size)

        # Gets the news index.
        print("Getting news indexes")
        for word, index in self.word_index_model.word_indexes.items():
            word_embedding = word_embedding_dict.get(word, None)
            if word_embedding is not None:
                embedding_matrix[index, :] = word_embedding
        return embedding_matrix

    @property
    def indexes(self):
        return self.headline_indexes

    def process_line(self, line):
        """ Takes in a line and processes it
        line : this is the line to process.
         """
        tokens = lines.split()
        date, ticker, indexes = tokens[0], tokens[1], tokens[2:]
        return date, ticker, indexes


class WordsIndexes:

    def __init__(self, filename):
        """ This takes in a file creates word to index and the corresponding index to word"""
        self.words = set()
        with open(filename, "r") as word_file:
            for line in word_file:
                self.words.add(line.strip().encode("utf-8").decode())

        special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

        self.index2word = {index: word for index,
                           word in enumerate(special_words + list(self.words))}

        self.word2index = {word: index for index,
                           word in self.index2word.items()}

    @property
    def indexed_words(self):
        """
        Returns the index: words.
        """
        return self.index2word

    def get_word_glove_vectors(self, glove_path = None):
        """ Takes in a word2index and returns a dictionary of the word and the corresponding matrix.
            word2index: This is the word2index values.
        """
        if glove_path == None:
            raise ValueError("glove_path must be a valid path")
        embedding_weights = {}
        with open(glove_path) as f:
            for line in f:
                try:
                    vals = line.split()
                    word = vals[0].encode("utf-8").decode()
                    if word in self.word_indexes:
                        coefs = np.asarray(vals[1:], dtype="float32")
                        coefs /= np.linalg.norm(coefs)
                        embedding_weights[word] = coefs
                except:
                    pass
        return embedding_weights


    @property
    def word_indexes(self):
        """
        Return the word: index.
        """
        return self.word2index


class CSVParser:

    def __init__(self):
        pass

    def parse(self, line):
        """ Parses a line into a line delimited by comma.
            line: this is the line of string to be parsed.
        """
        reader = csv.reader([line], skipinitialspace=True)
        for r in reader:
            return r


class HeadlineParser:

    def __init__(self):
        self.tokenizer = RegexpTokenizer("[\w']+")

    def tokenize(self, line):
        # return nltk.word_tokenize(line)
        return self.tokenizer.tokenize(line)

    def process(self, word):
        new_word = re.sub(r'(?<!\d)\.(?!\d)', '', word)
        new_word = new_word.replace("'", "")
        # new_word = new_word.replace("'s", "")
        return new_word

    def parse(self, headline):
        headline = self.process(headline)
        headline = self.tokenize(headline)
        headline = list(map(lambda x: x.lower(), headline))
        return headline
