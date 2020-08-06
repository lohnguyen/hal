import pickle

import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.optimizers import SGD


def process_data(lemmatizer, words, tags, X, y):
    X_train = []
    y_train = []

    for i, pattern in enumerate(X):
        bag = []
        pattern = [lemmatizer.lemmatize(w.lower()) for w in pattern]

        for w in words:
            if w in pattern:
                bag.append(1)
            else:
                bag.append(0)

        tag = [0] * len(tags)
        tag[tags.index(y[i])] = 1

        X_train.append(bag)
        y_train.append(tag)

    return np.array(X_train), np.array(y_train)


def train_model(X_train, y_train):
    input_shape = (len(X_train[0]),)
    model = Sequential()

    model.add(Dense(units=128, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(y_train[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)
    model.save('model')

    return model


def get_model(lemmatizer, words, tags, X, y):
    try:
        return load_model('model')
    except:
        X_train, y_train = process_data(lemmatizer, words, tags, X, y)
        model = train_model(X_train, y_train)
        return model
