import os
import sys
import csv
import nltk
import re
import numpy as np

from nltk.tokenize import RegexpTokenizer


class CSVParser:

    def __init__(self):
        pass

    @staticmethod
    def parse(line):
        """ Parses a line into a line delimited by comma.
            line: this is the line of string to be parsed.
        """
        reader = csv.reader([line], skipinitialspace=True)
        for r in reader:
            return r


class Padder:

    @staticmethod
    def padd(item, limit, pad_int=0, dimension=20, as_np_array=True):
        vectors_list = []
        sent_vecs = list(item)
        vectors_list.extend(sent_vecs)

        how_many_padding = limit - len(item)
        if as_np_array:
            paddings = [np.zeros(dimension) for i in range(how_many_padding)]
        else:
            paddings = [[pad_int] * dimension for i in range(how_many_padding)]

        reversed_vector_list = []
        reversed_vector_list.extend(paddings)
        reversed_vector_list.extend(reversed(sent_vecs))

        vectors_list.extend(paddings)
        return reversed_vector_list, vectors_list

    def padd_list(item, dimension, limit =10, np_arr=True):
        """ Takes in a list and pads it """
        if len(item) == limit:
            return item
        how_many = limit - len(item)
        vectors_list = []
        vectors_list.extend(item)
        paddings = [np.zeros(dimension) if np_arr else 0  for _ in range(how_many)]
        vectors_list.extend(paddings)
        return vectors_list




class Reverser:

    @staticmethod
    def reverse(item):
        """ Reverses a list. """
        vectors_list = []
        items = list(item)
        vectors_list.extend(items)
        reversed_items = list(reversed(items))
        return reversed_items, vectors_list




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
