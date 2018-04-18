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
from keras.callbacks import LambdaCallback


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

def prepare_X_y(data, num_char, sequences, time_step, char_to_idx_dict):

    dataX = []
    dataY = []

    for idx in range(0, num_char-sequences, time_step):
        sequence_in = data[idx:idx + sequences]
        next_chars = data[idx + sequences]
        dataX.append(sequence_in)
        dataY.append(next_chars)

    num_samples = len(dataX)

    return dataX, dataY, num_samples

def vectorize_X_y(dataX, sequences, num_chars, char_to_idx_dict, next_chars):

    X = np.zeros((len(dataX), sequences, num_chars), dtype=np.bool)
    y = np.zeros((len(dataX), num_chars), dtype=np.bool)

    for idx, pattern in enumerate(dataX):
        for jdx, char in enumerate(pattern):
            X[idx, jdx, char_to_idx_dict[char]] = 1
        y[idx, char_to_idx_dict[next_chars[idx]]] = 1

    return X, y

def model(X, y, num_chars):

    memory_units = 10
    dropout_rate = 0.3
    optimizer = RMSprop(lr=0.01)

    model = Sequential()
    model.add(LSTM(memory_units, input_shape=(sequences, num_chars))) #return_sequences=true # 1 label per timestep
    #model.add(Dropout(dropout_rate))
    model.add(Dense(num_chars, activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(data) - sequences - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        pattern = data[start_index: start_index + sequences]
        generated += pattern
        print('----- Generating with seed: "' + pattern + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, sequences, len(chars)))
            for t, char in enumerate(pattern):
                x_pred[0, t, char_to_idx_dict[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


if __name__ == '__main__':

    sequences = 25
    time_step = 1

    data, characters, num_chars, vocab_size = load_data('abc_test.txt')

    char_to_idx, idx_to_char = create_idx_dictionary(characters)

    dataX, dataY, num_samples = prepare_X_y(data, num_chars, sequences, time_step, char_to_idx)

    X, y = vectorize_X_y(dataX, sequences, num_chars, char_to_idx, dataY) #dayaY = next_char

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    model = model(X, y, num_chars)
    model.fit(X, y, epochs=20,
              batch_size=20,
              callbacks=[print_callback])













#
