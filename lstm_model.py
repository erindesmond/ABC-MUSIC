from __future__ import print_function
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
import pdb
import sys
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, TensorBoard
from keras.utils import plot_model
import graphviz
import pydot


def load_data(filename):
    '''Reads in a filename.txt
    Input: Text file of concatenated data
    Output: Data as text, unique vocabulary, num of char, num of vocab'''

    with tf.gfile.GFile(filename, 'r') as f:

        data = f.read()
        vocabulary = list(set(data))
        num_char = len(data)
        vocab_size = len(vocabulary)

    return data, vocabulary, num_char, vocab_size

def create_idx_dictionary(vocabulary):

    '''Creates index to vocabulary and vocabulary to index dictionaries.
    Input: The unique vocabulary of the data
    Output: Dictionaries storing the vocabulary and their index and vice versa'''

    char_to_idx = {char:idx for idx,char in enumerate(vocabulary)}
    idx_to_char = {idx:char for idx,char in enumerate(vocabulary)}

    return char_to_idx, idx_to_char

def prepare_X_y(data, num_char, sequences, char_to_idx_dict):

    '''Prepares the data for the neural network.
    Input: Data as txt, number of chars, desired sequence length, char to idx dictionary
    Output: dataX for "training" and dataY the "target" (which is one step ahead of dataX)'''

    dataX = []
    dataY = []

    for idx in range(0, num_char-sequences, 1):
        sequence_in = data[idx:idx + sequences]
        next_chars = data[idx + sequences]
        dataX.append(sequence_in)
        dataY.append(next_chars)


    return dataX, dataY

def vectorize_X_y(dataX, dataY, sequences, vocabulary, char_to_idx_dict):

    '''Converts dataX and dataY to vectors of boolean values with desired character set to 1
    Input: dataX and dataY, desired sequence, unique vocabulary, char to idx dictionary
    Output: X and y as boolean vectors.'''

    X = np.zeros((len(dataX), sequences, len(vocabulary)), dtype=np.bool)
    y = np.zeros((len(dataX), len(vocabulary)), dtype=np.bool)

    for idx, pattern in enumerate(dataX):
        for jdx, char in enumerate(pattern):
            X[idx, jdx, char_to_idx_dict[char]] = 1
        y[idx, char_to_idx_dict[dataY[idx]]] = 1

    return X, y

def model(X, y, vocabulary, sequences):

    '''Creates a Keras LSTM model
    Input: Vectorized X and y, unique vocabulary
    Output: A LSTM model'''

    memory_units = 100
    dropout_rate = 0.3
    optimizer = RMSprop(lr=0.01)

    model = Sequential()
    model.add(LSTM(memory_units, input_shape=(sequences, len(vocabulary)))) #return_sequences=true if you want more LSTMS
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(vocabulary), activation='softmax'))
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    #plot_model(model, to_file='model.png')

    return model

def sample_with_diversity(preds, diversity):

    '''Takes the array of predictions from the model that were generated from the sequence pattern,
    Takes the natural log of that distribution, and divides it by the diversity
    Exponentiate the results so that we're back to probabilities that don't sum to 1,
    Divide that by the sum of the exponentiated array so that they do sum to 1.

    Doing this with the diversity allows for more "randomness" in the model predictions.
    If the diversity == 1, the distribution is unchanged,
    If the diversity < 1, it makes the most probabable characters even more probable, reducing diversity
    If the diversity > 1, it makes the least probabable characters more probable, increasing diversity.

    Input: An array of the predicted values from the model, diversity list
    Output: The index of the largest number from the calculated probabilities, accounting for diversity'''

    preds = np.asarray(preds).astype('float64') # for better accuracy, but more memory intensive
    preds = np.log(preds) / diversity # take nat log of the preds, divide it by the diversity
    exp_preds = np.exp(preds) # the exponentiate the results, so that we're back to probabilities though no longer summing to 1
    preds = exp_preds / np.sum(exp_preds) # divide by the sum so that we do sum to 1
    probas = np.random.multinomial(1, preds, 1) # randomly grab from the new diverse distribution
    return np.argmax(probas) # return the idx of the largest number from the probabs

def on_epoch_end(epoch, logs):

    '''Generates predictive values from the LSTM model, in a range of diversities, translates those from
    the idx dictionary to a character, prints those to terminal in a specified range.
    Input: The epochs and logs for later use to callback in the model fit.
    Output: Callback in model fit to print generated text to terminal.'''

    data = load_data('abc_all.txt')[0]
    vocabulary = load_data('abc_all.txt')[1]
    sequences = 25
    idx_to_char_dict = create_idx_dictionary(vocabulary)[1]
    char_to_idx_dict = create_idx_dictionary(vocabulary)[0]
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = np.random.randint(0, len(data)-sequences - 1)
    # how far to deviate away from the probability distribution mean, further makes more random

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        # diversity explained above in sample_with_diversity function
        print('----- diversity:', diversity)

        # grab the first length of text from the random start index in len(sequences)
        music_generated = ''
        pattern = data[start_index: start_index + sequences]
        music_generated += pattern
        print('----- Generating with seed: "' + pattern + '"')
        sys.stdout.write(music_generated)

        for i in range(500):
            # in a print range of 500, create an empty probabilty matrix for filling later
            x_pred = np.zeros((1, sequences, len(vocabulary)))
            # from within the pattern generated above, grab a character
            for t, char in enumerate(pattern):
                # assign that character's index to 1 within the prediction matrix
                x_pred[0, t, char_to_idx_dict[char]] = 1.
            # call the predict method on that matrix for each of the characters grabbed (notice outside the loop)
            preds = model.predict(x_pred, verbose=0)[0] # the idx at the end to actually grab the prediction value from the method
            next_index = sample_with_diversity(preds, diversity) # plug that character into the diversity function to generate diversity or randomness
            next_char = idx_to_char_dict[next_index] # convert that idx to the associated character by using the dictionary

            music_generated += next_char # add that character to the music pattern generated from above
            pattern = pattern[1:] + next_char # drop the first character from the pattern, add the new character (taking a step forward)

            sys.stdout.write(next_char) # write the new character
            sys.stdout.flush() # force it to print on the screen

        print()

def test_model(trained_model):
    pass


    # print(trained_model.summary())  # As a reminder.
    # optimizer = RMSprop(lr=0.01)
    # # compile model
    # test_model = trained_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    #
    # return test_model


if __name__ == '__main__':

    training_data = 'abc_train.txt'
    testing_data = 'abc_test.txt'
    all_data = 'abc_all.txt'

    sequences = 25

    data, vocabulary, num_chars, vocab_size = load_data(all_data)

    char_to_idx, idx_to_char = create_idx_dictionary(vocabulary)

    dataX, dataY = prepare_X_y(data, num_chars, sequences, char_to_idx)

    X, y = vectorize_X_y(dataX, dataY, sequences, vocabulary, char_to_idx) #dayaY = next_char

    model = model(X, y, vocabulary, sequences)

    ''''''

    # test_data, test_vocabulary, test_num_chars, test_vocab_size = load_data(testing_data)
    #
    # test_char_to_idx, test_idx_to_char = create_idx_dictionary(test_vocabulary)
    #
    # test_dataX, test_dataY = prepare_X_y(test_data, test_num_chars, sequences, test_char_to_idx)
    #
    # X_test, y_test = vectorize_X_y(test_dataX, test_dataY, sequences, test_vocabulary, test_char_to_idx)


    batch_size = 100
    tensor_callback = tensorboard = TensorBoard(log_dir='./logs_two', batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    # model.fit(X, y, epochs=1,
    #           batch_size=batch_size,
    #           callbacks=[print_callback, tensor_callback])

    model.fit(X, y, epochs=50,
              batch_size=batch_size,
              callbacks=[print_callback, tensor_callback])

    # save_fname = 'trained_model'
    # # model.save_weights(save_fname + '.h5')
    # # ''''''
    # from keras.models import load_model
    # trained_model = load_model(save_fname)
    #
    # tested_model = test_model(trained_model)















#
