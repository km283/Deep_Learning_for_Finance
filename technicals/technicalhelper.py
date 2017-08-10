import os, sys
import numpy as np
sys.path.append(
    "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/")

from numpy import genfromtxt
from utils.model import Model

DATADIR = "/cs/home/un4/Documents/Dissertation/Data/Indexes/"
TRAIN = DATADIR + "training_index.csv"
VAL = DATADIR + "testing_index.csv"
TEST = DATADIR + "validation_index.csv"

class TechnicalData(Model):

    def __init__(self, filename):
        # Model.__init(self, filename)
        with np.load(filename, encoding="latin1") as data:
            # print(dir(data))
            self.technicals = data.f.arr_0[:,:,5:]
            self.technicals = self.technicals[:, ::-1, :]
            self.initializer()
            # assert self.technicals['arr_0'] == 800
        print("Matrix Shape",self.technicals.shape)
        print("Training Shape",self.train.shape)
        print("Validation Shape",self.val.shape)
        print("Testing Shape",self.test.shape)


    def initializer(self):
        train = TechnicalLoader(TRAIN).data
        val = TechnicalLoader(VAL).data
        test = TechnicalLoader(TEST).data

        self.train = self.technicals[train, :, :]
        self.val = self.technicals[val, :, :]
        self.test = self.technicals[test, :, :]


    def minibatch(self, item, batch_size):
        length = item.shape[0]
        print(length // batch_size)
        for i in range(length // batch_size):
            start = i * batch_size
            end = start + batch_size
            batch = self.technicals[start: end, :, :]
            yield batch

class TechnicalLoader:

    def __init__(self, filename, delimiter = ""):
        self._data = genfromtxt(filename, delimiter=delimiter)
        self._data = self._data.astype(np.int32)

    @property
    def data(self):
        return self._data


if __name__ == "__main__":
    train = TechnicalLoader(TRAIN).data
    print(train)
