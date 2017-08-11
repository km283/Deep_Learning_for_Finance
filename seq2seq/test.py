from WordIndexes import WordsIndexes
from Headline import Headline
from config import DATADIR, GLOVE, NEWS, DISTINCT



def main():
    print("this is lit")
    word_indexes = WordsIndexes(DISTINCT)
    headlines = Headline(NEWS, processed=False, word_index_model=word_indexes)
    embedded_matrix = headlines.get_embeded_matrix(glove_file=GLOVE)
    print(embedded_matrix[:4])
    print(headlines.indexes[: 3])

    # print(headlines.indexes)


if __name__ == "__main__":
    main()
