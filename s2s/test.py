from helper import *


DATADIR = "/Users/udokanwosu/Documents/Junk/ParseText/Seq2Seq/datafiles/"
DISTINCT = DATADIR + "hl_distinct.txt"
GLOVE = DATADIR + "glove.840B.300d.txt"
NEWS = DATADIR + "news.csv"


def main():
    print("this is lit")
    word_indexes = WordsIndexes(DISTINCT)
    # print(word_indexes.word2index)
    # print(word_indexes.word_indexes)
    # word_embedding_dict = word_indexes.get_word_glove_vectors(glove_path = GLOVE)
    # print(vectors)

    headlines = Headline(NEWS, processed=False, word_index_model=word_indexes)
    embedded_matrix = headlines.get_embeded_matrix(glove_file=GLOVE)
    print(embedded_matrix[:4])
    print(headlines.indexes[: 3])

    # print(headlines.indexes)


if __name__ == "__main__":
    main()
