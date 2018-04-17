import os
import tensorflow as tf
import re
import collections
import pdb

'cool'
# Consider removing some of the words from the text:
# perhaps anything that says X:, S:, or % Nottingham music database


# Concatenate all the abc files together
# I then manually split into 10% test and 90% train
# This resulted in 850 train songs and 94 test songs
def concatFiles():
    '''Takes a path that contains files to be concatenated, outputs one text file
    with all files together. Only run once!'''

    path = './nottingham-dataset/ABC_cleaned/'
    files = os.listdir(path)
    for idx, infile in enumerate(files):
        print ("File #" + str(idx) + "  " + infile)
    concat = ''.join([open(path + f).read() for f in files])
    with open("abc_all.txt", "w") as fo:
        fo.write(path + concat)


def read_measures(filename):
    with tf.gfile.GFile(filename, 'r') as f:

        data = f.read()
        print(data)
        measures = data.split(r"|")
        return measures
        #return f.read().replace('|', '<eos>').split()

def build_vocab(filename):
    data = read_measures(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def file_to_word_ids(filename, word_to_id):

    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_data():
    # get the data paths

    train_path = os.path.join('./', "abc_train.txt")
    test_path = os.path.join('./', "abc_test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, test_data, vocabulary, reversed_dictionary






if __name__ == "__main__":
    #concatFiles()
    train_data, test_data, vocabulary, reversed_dict = load_data()





    # f = open('abc_train.txt')
    #
    # for word in f.read().split():
    #     print(word)







    # def load_data():
    #     train_path = os.path.join(./ABC-LSTM/, "abc_train.txt")
    #     test_path = os.path.join(./ABC-LSTM/, "abc_test.txt")
    #
    #     word_to_id = build_vocab
    #     train_data = file_to_word_ids
    #     test_data =
    #     abc_vocab =
    #     reverse_dict =
