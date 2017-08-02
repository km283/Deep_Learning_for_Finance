import os
import sys
import csv
import nltk
import re
import numpy as np

from nltk.tokenize import RegexpTokenizer


class HeadlineEmbeddingModel:

    def __init__(self, filename, processed=True, word_index_model=None, with_information=False):
        """
        Reads the news headline and then returns a vector of those values.
        filename: this represents the file name of the file.
        wordmodel: this is an instance of the word class that returns the index2word and word2index.
        """
        self._word_embeddings = None
        self.word_index_model = word_index_model
        if processed:
            self.headline_indexes = self.processed_file(
                filename, with_information=with_information)
        else:
            self.headline_indexes = self.not_processed(
                filename, with_information=with_information)

    def word_embedding_dict(self):
        if self._word_embeddings is None:
            self._word_embeddings = self.word_index_model.get_word_glove_vectors()
        return self._word_embeddings

    def not_processed(self, filename, with_information=False):
        """ This function run when the file has not been processed
            filename: information filename.
            wordmodel: word model from class Word.
        """
        parser = HeadlineParser()
        with open(filename, "r") as headlines_file:
            indexed_headlines = []
            for line in headlines_file:
                items = CSVParser.parse(line)
                date = items[2]
                ticker = items[0]
                tokenized_headline = parser.parse(items[3])
                indexes = self.convert_line_to_index(
                    tokenized_headline)
                if with_information:
                    indexed_headlines.append((date, ticker, indexes))
                else:
                    indexed_headlines.append(indexes)
        return indexed_headlines

    def pad_sentence_vector(self, sentence_vector_list, padint=0, padding_length=54, dimension=300, reverse=False):
        """
            Pads the sentence to the representation of the sentence list.
        """
        # vector_list = [list(sentence_list)]
        vectors_list = []
        sent_vecs = list(sentence_vector_list)
        vectors_list.extend(sent_vecs)

        how_many_padding = padding_length - len(sent_vecs)
        paddings = [[padint] * dimension for i in range(how_many_padding)]

        reversed_vector_list = []
        reversed_vector_list.extend(paddings)
        reversed_vector_list.extend(reversed(sent_vecs))

        vectors_list.extend(paddings)

        return reversed_vector_list, vectors_list

    def get_vectors(self, headline, pad_how_much=5, length = 10):
        if not isinstance(headline, list):
            raise ValueError("Headline should be a list of string")
        np.random.seed(111)
        unk = np.random.randn(300)
        f = lambda x: self.word_embedding_dict().get(x, unk)
        vectors = list(map(f, headline))
        vectors = Padder.padd(vectors,
                              pad_how_much,
                              pad_int=0,
                              dimension=300,
                              as_np_array=True)

        return vectors

    def get_vectors_from_headline(self, headline, pad_len=53, dimension=300, reverse=False):
        """
        This takes in the dictionary of headlines and returns it sequentially.
        for example takes in "Donald Trump Laughs --> [[0.08980, ...], [0.08980, ...], [0.08980, ...]]
        headline: A list of strings.
        """
        if not isinstance(headline, list):
            raise ValueError("headline should be a list of strings")

        # vectors = list(map(lambda x : list(self.word_embedding_dict().get(x, "<UNK>")), headline))
        np.random.seed(1111)
        # unk_token = self.word_embedding_dict().get("<UNK>")
        unk = np.random.randn(300)
        f = lambda x: self.word_embedding_dict().get(x, unk)

        vectors = map(f, headline)

        if pad_len > 0:
            vectors = self.pad_sentence_vector(
                vectors, padint=0, padding_length=53, dimension=dimension, reverse=False)
        else:
            vectors = reversed(vectors), vectors
        return vectors

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

    def get_embeded_matrix(self):
        glove_size = 300
        # self.word_embedding_dict = self.word_index_model.get_word_glove_vectors(glove_file)
        # print(type(self.word_embedding_dict(glove_file)))
        embedding_matrix = np.random.randn(
            len(self.word_index_model.word_indexes), glove_size)
        # Gets the news index.
        # print("Getting news indexes")
        for word, index in self.word_index_model.word_indexes.items():
            word_embedding = self.word_embedding_dict().get(word, None)
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


class News:

    def __init__(self, filename, word_dict=None):
        """ This just parses the news 
            filename to parse
        """
        self.filename = filename
        self.tokenizer = HeadlineParser()
        self.index = 0
        self.last_batch_index = 0
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

    def parse_news_line(self, line):
        items = CSVParser.parse(line)
        date = items[2]
        ticker = items[0]
        headline = self.tokenizer.parse(items[3])
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
            batch = map(self.parse_news_line, self.newsfilelines[
                        start_i: start_i + batch_size])
            yield batch

    def reset_minibatch(self):
        self.last_batch_index = 0

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


class WordsIndexes:

    def __init__(self, filename, glove_file=None):
        """ This takes in a file creates word to index and the corresponding index to word"""
        self.glove_path = glove_file
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

    def get_word_glove_vectors(self):
        """ Takes in a word2index and returns a dictionary of the word and the corresponding matrix.
            word2index: This is the word2index values.
        """
        if self.glove_path == None:
            raise ValueError(
                "glove_path must be a valid path \nPath given is {}".format(glove_path))
        embedding_weights = {}
        with open(self.glove_path) as f:
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
