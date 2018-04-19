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


def load_data(filename):

    with tf.gfile.GFile(filename, 'r') as f:

        data = f.read()
        characters = list(set(data))
        num_char = len(data)
        vocab_size = len(characters)

    return data, characters, num_char, vocab_size

def create_idx_dictionary(characters):

        char_to_idx = {char:idx for idx,char in enumerate(characters)}
        idx_to_char = {idx:char for idx,char in enumerate(characters)}

        return char_to_idx, idx_to_char

def prepare_X_y(data, num_char, sequences, char_to_idx_dict):

    dataX = []
    dataY = []

    for idx in range(0, num_char-sequences, 1):
        sequence_in = data[idx:idx + sequences]
        next_chars = data[idx + sequences]
        dataX.append(sequence_in)
        dataY.append(next_chars)

    num_samples = len(dataX)

    return dataX, dataY, num_samples

def vectorize_X_y(dataX, sequences, characters, char_to_idx_dict, next_chars):

    X = np.zeros((len(dataX), sequences, len(characters)), dtype=np.bool)
    y = np.zeros((len(dataX), len(characters)), dtype=np.bool)

    for idx, pattern in enumerate(dataX):
        for jdx, char in enumerate(pattern):
            X[idx, jdx, char_to_idx_dict[char]] = 1
        y[idx, char_to_idx_dict[next_chars[idx]]] = 1

    return X, y

def model(X, y, characters):

    memory_units = 10
    dropout_rate = 0.3
    optimizer = RMSprop(lr=0.01)

    model = Sequential()
    model.add(LSTM(memory_units, input_shape=(sequences, len(characters)))) #return_sequences=true if you want more LSTMS
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(characters), activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

def sample_from_distribution(preds, diversity):
    # helper function to sample an index from a probability array
    # essentially, this normalizes the predicted probablility distribution numbers
    # by dividing them by the diversity argument. This further randomizes the choosing
    # process so that we don't generate the exact same text that we trained on

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):

    data = load_data('abc_test.txt')[0]
    characters = load_data('abc_test.txt')[1]
    sequences = 25
    idx_to_char_dict = create_idx_dictionary(characters)[1]
    char_to_idx_dict = create_idx_dictionary(characters)[0]
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = np.random.randint(0, len(data)-sequences - 1)
    # how far to deviate away from the probability distribution mean, further makes more random

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        # grab the first length of text from the random start index in len(sequences)
        music_generated = ''
        pattern = data[start_index: start_index + sequences]
        music_generated += pattern
        print('----- Generating with seed: "' + pattern + '"')
        sys.stdout.write(music_generated)

        for i in range(400):
            x_pred = np.zeros((1, sequences, len(characters)))
            for t, char in enumerate(pattern):
                x_pred[0, t, char_to_idx_dict[char]] = 1.


            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample_from_distribution(preds, diversity)
            next_char = idx_to_char_dict[next_index]

            music_generated += next_char
            pattern = pattern[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


if __name__ == '__main__':

    sequences = 25

    data, characters, num_chars, vocab_size = load_data('abc_all.txt')

    char_to_idx, idx_to_char = create_idx_dictionary(characters)

    dataX, dataY, num_samples = prepare_X_y(data, num_chars, sequences, char_to_idx)

    X, y = vectorize_X_y(dataX, sequences, characters, char_to_idx, dataY) #dayaY = next_char

    model = model(X, y, characters)
    batch_size = 100

    tensor_callback = tensorboard = TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


    model.fit(X, y, epochs=20,
              batch_size=batch_size,
              callbacks=[print_callback, tensor_callback])













#
