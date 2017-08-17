import os, sys



class Errorlog:

    def __init__(self, filename, a_label, b_label):
        with open(filename, "w") as error_file:
            self.error_file = errorfile
            self.error_file.write("{}, {}\n".format(a_label, b_label))

    def log(self, a, b):
        self.error_file.write("{}, {}\n".format(a, b))
