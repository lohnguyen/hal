import pickle
import os

import nltk


def dump_pickle_variables(words, tags, X, y):
    if not os.path.isdir('variables'):
        os.makedirs('variables')

    with open('variables/words.pickle', 'wb') as file:
        pickle.dump(words, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open('variables/tags.pickle', 'wb') as file:
        pickle.dump(tags, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open('variables/X.pickle', 'wb') as file:
        pickle.dump(X, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open('variables/y.pickle', 'wb') as file:
        pickle.dump(y, file, protocol=pickle.HIGHEST_PROTOCOL)


def preprocess_data(intents, lemmatizer):
    try:
        words = pickle.load(open('variables/words.pickle', 'rb'))
        tags = pickle.load(open('variables/tags.pickle', 'rb'))
        X = pickle.load(open('variables/X.pickle', 'rb'))
        y = pickle.load(open('variables/y.pickle', 'rb'))
        return words, tags, X, y
    except:
        words = []
        tags = []
        X = []
        y = []

        for intent in intents:
            for pattern in intent['patterns']:
                tokenized_words = nltk.word_tokenize(pattern)
                words.extend(tokenized_words)
                X.append(tokenized_words)
                y.append(intent['tag'])

                if intent['tag'] not in tags:
                    tags.append(intent['tag'])

        words = [lemmatizer.lemmatize(w.lower())
                 for w in words if (w != '?' and w != '!')]
        words = sorted(list(set(words)))
        tags = sorted(list(set(tags)))

        dump_pickle_variables(words, tags, X, y)

        return words, tags, X, y
