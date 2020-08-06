import json
import pickle
import random
import os
import sys
import shutil

import nltk
import numpy as np

from preprocessing import preprocess_data
from model import get_model
from hal import print_hal, print_line

lemmatizer = nltk.stem.WordNetLemmatizer()


def bag_of_words(inp, words):
    bag = [0] * len(words)

    inp = nltk.word_tokenize(inp)
    inp = [lemmatizer.lemmatize(w.lower()) for w in inp]

    for word in inp:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1

    return np.array([bag])


def get_response(model, inp, name, intents, words, tags):
    bag = bag_of_words(inp, words)
    predictions = model.predict(bag)
    result_index = np.argmax(predictions)
    tag = tags[result_index]
    responses = []

    for intent in intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            break

    response = random.choice(responses).replace('usrName', name)

    if tag == 'bye':
        print_line(response)
        exit()

    return response


def main():
    name = input("State your first name to start: ")
    intents = json.load(open('intents.json'))
    words, tags, X, y = preprocess_data(intents, lemmatizer)
    model = get_model(lemmatizer, words, tags, X, y)

    print_hal()

    while True:
        inp = input(name + ": ")
        response = get_response(model, inp, name, intents, words, tags)

        print_line(response)


if __name__ == "__main__":
    os.system('clear')

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        if os.path.isdir('model'):
            shutil.rmtree('model')

        if os.path.isdir('variables'):
            shutil.rmtree('variables')

    main()
