import re
import os, sys


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
