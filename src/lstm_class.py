import tensorflow as tf
import numpy as np
import sys
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback, TensorBoard

class ABC():

    def __init__(self, data, sequences, epochs, batch_size):
        self.data = data
        self.sequences = sequences
        self.epochs = epochs
        self.batch_size = batch_size

    def apply_functions(self):

        self.load_data()
        self.create_idx_dictionary()
        self.prepare_X_y()
        self.vectorize_X_y()
        self.model()
        self.fit_model()

    def load_data(self):
        '''Reads in a filename.txt
        Input: Text file of concatenated data
        Output: Data as text, unique vocabulary, num of char, num of vocab'''

        with tf.gfile.GFile(self.data, 'r') as f:

            self.abc = f.read()
            self.vocabulary = list(set(self.abc))
            self.num_char = len(self.abc)
            self.vocab_size = len(self.vocabulary)

    def create_idx_dictionary(self):

        '''Creates index to vocabulary and vocabulary to index dictionaries.
        Input: The unique vocabulary of the data
        Output: Dictionaries storing the vocabulary and their index and vice versa'''

        self.char_to_idx_dict = {char:idx for idx,char in enumerate(self.vocabulary)}
        self.idx_to_char_dict = {idx:char for idx,char in enumerate(self.vocabulary)}


    def prepare_X_y(self):

        '''Prepares the data for the neural network.
        Input: Data as txt, number of chars, desired sequence length, char to idx dictionary
        Output: dataX for "training" and dataY the "target" (which is one step ahead of dataX)'''

        self.dataX = []
        self.dataY = []

        for idx in range(0, self.num_char - self.sequences, 1):
            sequence_in = self.abc[idx:idx + self.sequences]
            next_chars = self.abc[idx + self.sequences]
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

    def model(self):

        '''Creates a Keras LSTM model
        Input: Vectorized X and y, unique vocabulary
        Output: A LSTM model'''

        memory_units = 100
        dropout_rate = 0.3
        rmsprop = RMSprop(lr=0.001) # found minimum at around 20
        #adam = Adam(lr=0.001) #rmsprop worked better

        self.model = Sequential()
        self.model.add(LSTM(memory_units, input_shape=(self.sequences, len(self.vocabulary)), return_sequences=True))
        self.model.add(LSTM(memory_units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(len(self.vocabulary), activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

    def fit_model(self):

        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        tensor_callback = TensorBoard(log_dir='./logs',
        batch_size=self.batch_size, write_graph=True, write_grads=True, write_images=True)

        self.model.fit(self.X, self.y, epochs=self.epochs,
        batch_size=self.batch_size, callbacks=[tensor_callback, print_callback])

        return self.model

    def sample_with_diversity(self, diversity):

        '''Takes the array of predictions from the model that were generated from the sequence pattern,
        Takes the natural log of that distribution, and divides it by the diversity
        Exponentiate the results so that we're back to probabilities that don't sum to 1,
        Divide that by the sum of the exponentiated array so that they do sum to 1.
        Doing this with the diversity allows for more "randomness" in the model predictions.
        If the diversity == 1, the distribution is unchanged,
        If the diversity < 1, it makes the most probabable characters even more probable, reducing diversity
        If the diversity > 1, it makes the least probabable characters more probable, increasing diversity.
        Input: An array of the predicted values from the model, diversity list
        Output: The index of the largest number from the calculated probabilities, accounting for diversity.'''

        self.predictions = np.asarray(self.predictions).astype('float64') # for better accuracy, but more memory intensive
        self.predictions = np.log(self.predictions) / diversity # take nat log of the preds, divide it by the diversity
        exp_preds = np.exp(self.predictions) # the exponentiate the results, so that we're back to probabilities though no longer summing to 1
        self.predictions = exp_preds / np.sum(exp_preds) # divide by the sum so that we do sum to 1
        probas = np.random.multinomial(1, self.predictions, 1) # randomly grab from the new diverse distribution
        self.probas = np.argmax(probas) # return the idx of the largest number from the probabs

        return self.probas

    def on_epoch_end(self, epoch, logs):

        print()
        print('\n----- Generating text after Epoch: %d' % epoch)

        start_idx = np.random.randint(0, len(self.abc)-self.sequences - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('\n----- For diversity:', diversity)

            self.music_generated = ''
            pattern = self.abc[start_idx: start_idx + self.sequences]
            self.music_generated += pattern

            print('----- Generating with seed: "' + pattern + '"''\n')
            sys.stdout.write(self.music_generated)

            for i in range(500):
                x_pred = np.zeros((1, self.sequences, len(self.vocabulary)))

                for x, char in enumerate(pattern):
                    x_pred[0, x, self.char_to_idx_dict[char]] = 1.

                self.predictions = self.model.predict(x_pred, verbose=2)[0]
                next_idx = self.sample_with_diversity(diversity)
                next_char = self.idx_to_char_dict[next_idx]

                self.music_generated += next_char
                pattern = pattern[1:] + next_char

                sys.stdout.write(next_char) # write the new character
                sys.stdout.flush()

            print()


if __name__ == '__main__':

    # training_data = '/Users/erindesmond/Documents/abc_lstm/data/abc_train.txt'
    # testing_data = '/Users/erindesmond/Documents/abc_lstm/data/abc_test.txt'
    # classical_test = '/Users/erindesmond/Documents/abc_lstm/data/classical_test.txt'
    irish = '/Users/erindesmond/Documents/music_capstone/data/abc_all.txt'
    bach = '/Users/erindesmond/Documents/music_capstone/data/bach.rtf'
    enya = '/Users/erindesmond/Documents/music_capstone/data/enya.rtf'
    mj = '/Users/erindesmond/Documents/music_capstone/data/mj.rtf'
    everyone = '/Users/erindesmond/Documents/music_capstone/data/all_together.rtf'

    sequences = 25
    epochs = 5
    batch_size = 100

    model = ABC(mj, sequences, epochs, batch_size).apply_functions()


    ''''''











#











#
