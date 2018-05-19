import tensorflow as tf
import numpy as np

class PrepareData():

    def __init__(self, data, sequences):
        self.data = data
        self.sequences = sequences
        self.vocabulary = None
        self.num_char = None
        self.vocab_size = None
        self.char_to_idx_dict = {}
        self.idx_to_char_dict = {}
        self.dataX = []
        self.dataY = []
        self.X = []
        self.y = []

    def apply_prep_methods(self):

        self.load_data()
        self.create_idx_dictionary()
        self.prepare_X_y()
        self.vectorize_X_y()

    def load_data(self):
        '''Reads in a filename.txt'''

        with tf.gfile.GFile(self.data, 'r') as f:

            self.data = f.read()
            self.vocabulary = list(set(self.data))
            self.num_char = len(self.data)
            self.vocab_size = len(self.vocabulary)

    def create_idx_dictionary(self):

        '''Creates index to vocabulary and vocabulary to index dictionaries'''

        self.char_to_idx_dict = {char:idx for idx,char in enumerate(self.vocabulary)}
        self.idx_to_char_dict = {idx:char for idx,char in enumerate(self.vocabulary)}

    def prepare_X_y(self):

        '''Prepares the data for the neural network.
        Input: Data as txt, number of chars, desired sequence length, char to idx dictionary
        Output: dataX for "training" and dataY the "target" (which is one step ahead of dataX)'''

        for idx in range(0, self.num_char - self.sequences, 1):
            sequence_in = self.data[idx:idx + self.sequences]
            next_chars = self.data[idx + self.sequences]
            self.dataX.append(sequence_in)
            self.dataY.append(next_chars)

    def vectorize_X_y(self):

        '''Converts dataX and dataY to vectors of boolean values with desired character set to 1
        Input: dataX and dataY, desired sequence, unique vocabulary, char to idx dictionary
        Output: X and y as boolean vectors.'''


        self.X = np.zeros((len(self.dataX), self.sequences, len(self.vocabulary)), dtype=np.bool)
        self.y = np.zeros((len(self.dataX), len(self.vocabulary)), dtype=np.bool)

        for idx, pattern in enumerate(self.dataX):
            for jdx, char in enumerate(pattern):
                self.X[idx, jdx, self.char_to_idx_dict[char]] = 1
            self.y[idx, self.char_to_idx_dict[self.dataY[idx]]] = 1


if __name__ == '__main__':

    training_data = '../data/abc_train.txt'
    testing_data = '../data/abc_test.txt'
    all_data = '../data/abc_all.txt'
    classical_test = '../data/classical_test.txt'
    bach = '../data/bach.rtf'
    sequences = 25

    prep_data = PrepareData(bach, sequences)









    ''''''
