import os, sys
import re
import nltk
import numpy as np

from utils.helper import HeadlineParser, CSVParser

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
        unk = self.word_index_model.word_indexes.get("<UNK>")
        return list(map(lambda x: self.word_index_model.word_indexes.get(x, unk), line))

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
