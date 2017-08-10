import sys, os.path
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.model import Model
from utils.helper import CSVParser, Padder, Reverser

class DayHeadlineModel(Model):

    def __init__(self, filename):
        """
        Constructor, filename.
        """
        print("Reading filename" + filename)
        Model.__init__(self, filename)
        self.date_dict = self.group_by_day()

    def get_days(self):
        """ This will return all of the days. """
        days = list(map(lambda x : x.keys(), self.date_dict))

    def group_by_day(self):
        """
            This groups the headlines by day.
        """
        date_dict = {}
        for items in self.lines:
            values = CSVParser.parse(items)
            date = values[0]
            ticker = values[1]
            # vectors = list(map(float, values[2].split()))
            vectors = np.array(values[2].split()).astype(np.float)
            date_ticker = date_dict.get(ticker, None)
            if date_ticker == None:
                day_vectors = {}
                vectors_list = []
                # vectors_list.append(np.array(vectors))
                vectors_list.append(vectors)
                day_vectors[date] = vectors_list
                date_dict[ticker] = day_vectors
            else:
                date_ticker_vector = date_ticker.get(date, None)
                if date_ticker_vector == None:
                    vectors_list = []
                    vectors_list.append(vectors)
                    date_dict[ticker][date] = vectors_list
                else:
                    date_dict[ticker][date].append(vectors)
        return date_dict

    def __len__(self):
        """
        This returns the len of the items.
        """
        if self.length == None:
            length = 0
            for k, v in self.date_dict.items():
                length += len(v.values())
            self.length = length
        return self.length

    def get_stats(self):
        """
        Returns basic statistics of the file
        """
        counts = []
        # counts = list(map(len, self.date_dict.values()))
        # counts = np.array(counts)
        # print(set(counts))
        # return np.mean(counts), np.median(counts)

    def max_day_headlines(self):
        """
        This returns the max headline of the item list.
        """
        max_value = 0
        for k, v in self.date_dict.items():
            # print(len(v.values()))
            max_value = max(max_value, len(max(v.values(), key=len)))
        return max_value

    def min_day_headlines(self):
        # for _, values in self.date_dict:
        min_value = 10000
        for k, v in self.date_dict.items():
            min_value = min(min_value, len(min(v.values(), key=len)))
        return min_value

    # def autoencoder_minibatch(self, batch_size):
    #     """
    #     Gets the minibatch and then send the sequence length.
    #     """

    def minibatch(self, batch_size, pad=True, pad_how_much=12, full_information=False, dimension=400):
        """ Gets minibatch. """
        # TODO: Get minibatch function to also send the sequence len of the file that is being sent.
        # The return value should be (batch, seq_array).
        item_batches = []
        if full_information:
            for k, v in self.date_dict.items():
                for k1, v1 in v.items():
                    if pad:
                        item = Padder.padd(v1, pad_how_much, pad_int=0, dimension=dimension)
                    else:
                        item = Reverser.reverse(v1)
                    item_batches.append((k, k1, item))
        else:
            for k, v in self.date_dict.items():
                for item in v.values():
                    if pad:
                        item = Padder.padd(item, pad_how_much, pad_int=0, dimension=dimension)
                    else:
                        item = Reverser.reverse(item)
                    item_batches.append(item)
            item_batches = sorted(item_batches, key=lambda x : len(x[0]), reverse = True)

        # print("Headline per day", len(item_batches[0]))
        # print(list(map(len, item_batches)))
        # item_batches = sorted(item_batches, key=len)
        for i in range(0, self.__len__() // batch_size):
            start_i = i * batch_size
            batch = item_batches[start_i: start_i + batch_size]
            yield batch
