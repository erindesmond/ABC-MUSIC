import tensorflow as tf
import numpy as np
import sys
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard, LambdaCallback
from prepare_data import PrepareData


class SaveModel():

    def __init__(self, sequences, vocabulary, batch_size, epochs, X, y):

        self.sequences = sequences
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.epochs = epochs
        self.X = X
        self.y = y

    def apply_model_methods(self):

        self.model()
        self.fit_model()

    def model(self):
        '''Creates a Keras LSTM model'''

        memory_units = 100
        dropout_rate = 0.3
        rmsprop = RMSprop(lr=0.001) 
        #adam = Adam(lr=0.001) #rmsprop worked better

        self.model = Sequential()
        self.model.add(LSTM(memory_units, input_shape=(self.sequences, len(self.vocabulary)), return_sequences=True))
        self.model.add(LSTM(memory_units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(len(self.vocabulary), activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

    def fit_model(self):
        '''Fits the LSTM to X and y'''

        tensor_callback = TensorBoard(log_dir='./logs',
        batch_size=self.batch_size, write_graph=True, write_grads=True, write_images=True)

        self.model.fit(self.X, self.y, epochs=self.epochs,
        batch_size=self.batch_size, callbacks=[tensor_callback])

        self.model.save('trained_model.h5')


if __name__ == '__main__':

    training_data = '../data/abc_train.txt'
    testing_data = '../data/abc_test.txt'
    all_data = '../data/abc_all.txt'
    classical_test = '../data/classical_test.txt'
    bach = '../data/bach.rtf'
    sequences = 25
    epochs = 40
    batch_size = 100

    prepared_data = PrepareData(testing_data, sequences)
    prepared_data.apply_prep_methods()

    vocabulary = prepared_data.vocabulary
    X = prepared_data.X
    y = prepared_data.y

    best_model = SaveModel(sequences, vocabulary, batch_size, epochs, X, y)
    best_model.apply_model_methods()








''''''
