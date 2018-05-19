def test_rnn(trained_model, dataX, dataY, char_to_idx_dict, batch_size):
    '''
    Compares the generated text prediction to known subsequent text
    Parameters
    ----------
    model_path: filepath to fitted/trained keras RNN model saved as .h5
    text: STR - text to predict on
    word_indices: DICT - vocab dictionary for training data where keys are
        words (str) and values are indices (int)
    batch_size: INT - batch size for updating weights
    Returns
    -------
    None
    '''
    model = load_model(trained_model)
    indices_word = dict((v, k) for k, v in word_indices.items())

    X, y = _vectorize_text(tokens, word_indices, seq_length=SEQ_LENGTH,
                           step=STEP)
    predict = model.predict(X, batch_size=batch_size)

    predict_ = tokens_to_text(predict, indices_word, precedes_unknown_token)
    y_ = tokens_to_text(y, indices_word, precedes_unknown_token)

    print('\nActual text')
    print(y_)

    print('\nNew text')
    print(predict_)


# load saved weights
from keras.models import load_model
model = load_model(current + '/' + '30_epochs.h5')
print(model.summary())  # As a reminder.

# compile model
model.compile(loss='categorical_crossentropy', # binary_crossentropy
              optimizer='rmsprop',
              metrics=['accuracy'])

# test model
score = model.evaluate_generator(test_generator, nb_validation_samples // batch_size + 1) # read about size thing on stack exchange
metrics = model.metrics_names
# model.metrics_names to get score labels
print('{} = {}'.format(metrics[0],score[0]))
print('{} = {}'.format(metrics[1],score[1]))
y_pred = model.predict_generator(test_generator, nb_validation_samples // batch_size + 1, verbose = 1)

# plot predictions (probability of class0, class1)
plot_predictions(y_pred,'figs/' + savename + '_predictions')
