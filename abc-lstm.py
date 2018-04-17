import tensorflow as tf
import numpy as np
import pdb

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed


def load_data(filename):

    with tf.gfile.GFile(filename, 'r') as f:

        data = f.read()
        characters = list(set(data))
        vocab_size = len(characters)

    return data, characters, vocab_size

def create_idx_dictionary(characters):

        char_to_idx = {char:idx for idx,char in enumerate(characters)}
        idx_to_char = {idx:char for idx,char in enumerate(characters)}

        return char_to_idx, idx_to_char

def prepare_X_y(data, len_sequences, num_sequences, num_features, char_to_idx_dict):

    X = np.zeros((len(data)//len_sequences, len_sequences, num_features))
    y = np.zeros((len(data)//len_sequences, len_sequences, num_features))

    for i in range(0, int(num_sequences)):

        # assign the sequence in chosen length from data to X_sequence
        # indicate the index for that sequence in the dict from above, assign to list
        X_sequence = data[i*len_sequences:(i+1)*len_sequences]
        X_sequence_idx = [char_to_idx_dict[value] for value in X_sequence]

        # Create empty matrix of zeros for the input sequences to arrange by idx
        # This allows us to 'one-hot-encode' based on where that sequence happened
        # If that sequence is in that index, set equal to 1
        input_sequence = np.zeros((len_sequences, num_features))
        for j in range(len_sequences):
            input_sequence[j][X_sequence_idx[j]] = 1.

    X[i] = input_sequence

    # target sequence will be same as X, but taking one step forward. This is to
    # set the target by shifting the corresponding input sequence
    # by one character. (So it's predicting on the next character)
    # indicate the index for that sequence from the dict from above, assign to list
    y_sequence = data[i*len_sequences+1:(i+1)*len_sequences+1]
    y_sequence_idx = [char_to_idx_dict[value] for value in y_sequence]

    # here we set the target sequence by indexing into y with the y_sequence (which is
    # a step ahead of X, remember)
    target_sequence = np.zeros((len_sequences, num_features))
    for j in range(len_sequences):
        target_sequence[j][y_sequence_idx[j]] = 1.
    y[i] = target_sequence

    return X, y

def generate_text(model, length):
    idx = [np.random.randint(vocab_size)]
    y_char = [idx_to_char_dict[idx[-1]]]
    X = np.zeros((1, length, vocab_size))

    for i in range(length):

        X[0, i, :][idx[-1]] = 1
        print(idx_to_char_dict[idx[-1]], end="")
        idx = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(idx_to_char_dict[idx[-1]])
    return ('').join(y_char)

if __name__ == '__main__':

    data, characters, vocab_size = load_data('abc_all.txt')
    char_to_idx_dict, idx_to_char_dict = create_idx_dictionary(characters)

    len_sequences = 10
    num_sequences = int(len(data) / len_sequences)
    num_features = vocab_size

    X, y = prepare_X_y(data, len_sequences, num_sequences, num_features, char_to_idx_dict)


    dropout = 0.3
    internal_size = 2


    model = Sequential()

    number_of_layers = 2
    for layer in range(number_of_layers - 1):

        model.add(LSTM(internal_size,input_shape=(None, vocab_size),
                       return_sequences=True))

    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(vocab_size), kernel_initializer='random_uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    batch_size = 10
    generate_length = 50

    num_epoch = 0
    while True:
        print('\n\n')
        model.fit(X, y, batch_size=batch_size, verbose=1, epochs=1)
        num_epoch += 1
        generate_text(model, generate_length)

        if num_epoch % 10 == 0:
            model.save_weights('checkpoint_{}_epoch{}.hdf5'.format(internal_size, num_epoch))
