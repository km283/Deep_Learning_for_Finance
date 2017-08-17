import os, sys
import numpy as np
import pandas as pd
sys.path.append(
    "/cs/home/un4/Documents/Dissertation/Project/ComputationalInvesting/")

from numpy import genfromtxt
from utils.model import Model

DATADIR = "/cs/home/un4/Documents/Dissertation/Data/Indexes/"
TRAIN = DATADIR + "training_index.csv"
TEST = DATADIR + "testing_index.csv"
VAL = DATADIR + "validation_index.csv"
TECH_LABELS = "/cs/home/un4/Documents/Krystof_Folder/technicals_final.csv"

class TechnicalData(Model):

    def __init__(self, filename):
        # Model.__init(self, filename)
        self.initializer(filename)
            # assert self.technicals['arr_0'] == 800
        # print("Matrix Shape",self.technicals.shape)
        print("Training Shape",self.train[1].shape)
        print("Validation Shape",self.val[1].shape)
        print("Testing Shape",self.test[1].shape)

    def initializer(self, filename):
        with np.load(filename, encoding="latin1") as data:
            # print(dir(data))
            technicals = data.f.arr_0[:,:,5:]
            technicals = technicals[:, ::-1, :]
            labels = Labels(TECH_LABELS).labels
            print("Full Shape {}".format(technicals.shape))

            print(TRAIN)
            print(VAL)
            print(TEST)

            train = TechnicalLoader(TRAIN).data
            val = TechnicalLoader(VAL).data
            test = TechnicalLoader(TEST).data

            self.train = labels.iloc[train], technicals[train, :, :]
            self.val = labels.iloc[val], technicals[val, :, :]
            # print(dir(self.val[0]))
            # print(self.val[0].values.tolist())
            # sys.exit(1)
            self.test = labels.iloc[test], technicals[test, :, :]


    def minibatch(self, item, batch_size):
        labels = item[0]
        values = item[1]
        length = values.shape[0]
        # print(length // batch_size)
        for i in range(length // batch_size):
            start = i * batch_size
            end = start + batch_size
            batch_labels = labels[start: end].values.tolist()
            batch = values[start: end, :, :]
            yield batch_labels,  batch

class TechnicalLoader:

    def __init__(self, filename, delimiter = ""):
        self._data = genfromtxt(filename, delimiter=delimiter)
        self._data = self._data.astype(np.int32)

    @property
    def data(self):
        return self._data

class Labels:
    def __init__(self, filename):
        df = pd.read_csv(filename, sep=",", header=None)
        self.labels = df[df.columns[0:2]]


if __name__ == "__main__":
    # train = TechnicalLoader(TRAIN).data
    # print(train)
    # l = Labels(TECH_LABELS).labels[1:5]
    # print(dir(l))
    # f = "/cs/home/un4/Documents/Dissertation/Data/technical_list_of_arrays.npz"
    # t = TechnicalData(f)
    # for i in t.minibatch(t.train, 2):
    #     print(i)
    #     sys.exit(1)
    pass
