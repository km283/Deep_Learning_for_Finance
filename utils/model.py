import sys, os.path 
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

class Model:

    def __init__(self, filename):
        """
        Constructor, filename.
        """
        self.lines = None
        self.length = None
        with open(filename, "r") as day_headline_file:
            self.lines = day_headline_file.readlines()

    def __len__(self):
        """ This returns the len of the file. """
        return len(self.lines)

    def max_headline_count(self):
        """
        returns the max headline count
        """
        raise NotImplementedError("You need to implement this feature")

    def minibatch(self, batch_size):
        raise NotImplementedError("This method needs to be implemented") 