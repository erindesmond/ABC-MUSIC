import numpy as np
import sys
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import RMSprop, Adam

from prepare_data import PrepareData

class GenerateMusic():

    def __init__(self, data, saved_model, sequences, vocabulary, char_to_idx_dict, idx_to_char_dict, predictions, probabilities):
        self.data = data
        self.saved_model = saved_model
        self.sequences = sequences
        self.vocabulary = vocabulary
        self.char_to_idx_dict = char_to_idx_dict
        self.idx_to_char_dict = idx_to_char_dict
        self.predictions = None
        self.probabilities = None

    def apply_predictive_methods(self):

        self.compile_model()
        self.produce_and_print_results()

    def compile_model(self):
        '''Creates a Keras LSTM model'''

        rmsprop = RMSprop(lr=0.01) # found minimum at around 20
        adam = Adam(lr=0.001) #rmsprop worked better

        self.model = load_model(self.saved_model)
        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

    def sample_with_diversity(self, diversity):

        '''Takes the array of predictions from the model that were generated from the sequence pattern,
        Takes the natural log of that distribution, and divides it by the diversity.
        Exponentiate the results so that we're back to probabilities that don't sum to 1,
        Divide that by the sum of the exponentiated array so that they do sum to 1.

        Doing this with the diversity allows for more "randomness" in the model predictions.
        If the diversity == 1, the distribution is unchanged,
        If the diversity < 1, it makes the most probabable characters even more probable, reducing diversity
        If the diversity > 1, it makes the least probabable characters more probable, increasing diversity'''

        self.predictions = np.asarray(self.predictions).astype('float64') # for better accuracy, but more memory intensive
        self.predictions = np.log(self.predictions) / diversity # take nat log of the preds, divide it by the diversity
        exp_preds = np.exp(self.predictions) # the exponentiate the results, so that we're back to probabilities though no longer summing to 1
        self.predictions = exp_preds / np.sum(exp_preds) # divide by the sum so that we do sum to 1
        probabilities = np.random.multinomial(1, self.predictions, 1) # randomly grab from the new diverse distribution
        probability_index = np.argmax(probabilities) # return the idx of the largest number from the probabs

        return probability_index

    def produce_and_print_results(self):
        '''Prints the predicted music pattern to the terminal'''

        print()
        print('\n----- Generating text after Epoch: %d' % epoch)

        start_idx = np.random.randint(0, len(self.data)-self.sequences - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('\n----- For diversity:', diversity)

            self.music_generated = ''
            pattern = self.data[start_idx: start_idx + self.sequences]
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

    irish = '../data/abc_all.txt'
    bach = '../data/bach.rtf'
    enya = '../data/enya.rtf'
    mj = '../data/mj.rtf'
    everyone = '../data/all_together.rtf'

    sequences = 25
    epochs = 40
    batch_size = 100


    prepared_data = PrepareData(testing_data, sequences)
    prepared_data.apply_prep_methods()

    data = prepared_data.data
    vocabulary = prepared_data.vocabulary
    X = prepared_data.X
    y = prepared_data.y
    char_to_idx_dict = prepared_data.char_to_idx_dict
    idx_to_char_dict = prepared_data.idx_to_char_dict

    saved_model = 'trained_model.h5'

    generated_music = GenerateMusic(data, saved_model, sequences, vocabulary, char_to_idx_dict, idx_to_char_dict, predictions=None, probabilities=None)
    generated_music.apply_predictive_methods()







    ''''''
