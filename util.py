# util.py
# set of utility functions to make things smoother
# includes: 
# loadSettings, saveTokenizer, loadTokenizer

from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import os
import cv2
import glob
import toml
import pickle
import math

# returns settings dictionary
# must be in same dir as settings.toml to work
def loadSettings():
	f = open("settings.toml")
	settings_content = f.read()
	f.close()
	return toml.loads(settings_content)


settings = loadSettings()

# returns 2 2d numpy arrays
# (input, answer) of tokens
# tokenizer: https://keras.io/preprocessing/text/
# note: a value must be specified for unknown words
# or else it will be ignored and arrays of incorrect size 
# returned
def loadTrainingData(path_to_data, tokenizer):

    raw_text  = open(path_to_data).read()
    raw_data  = open(path_to_data).readlines()

    oov_token = tokenizer.oov_token

    # creates a map for words to tokens
    # higher frequency words will have a smaller
    # token value
    # note: ensure char_level is set to false (should be default)
    tokenizer.fit_on_texts(raw_data)

    # convert raw text to arrays of words (sequences)
    raw_sequences = [text_to_word_sequence(line) for line in raw_data]
    tokens        = tokenizer.texts_to_sequences(raw_sequences)    

    # TODO look into a better way of determining max_word_count
    max_word_count = settings['tokenizer']['max_word_count']
    vocab          = len(tokenizer.word_counts)

    # pad to max size
    # todo: possibly set limit for this
    # convert to float and normalize
    tokens = np.array([sub_tokens + [tokenizer.oov_token] * (max_word_count - len(sub_tokens)) for sub_tokens in tokens])#.astype('float32') / vocab

    # its expected that [::2] is a user question and [1::2] is the bots answer
    # split array in two for input/anwers
    # note: arr[start:stop:step]
    print("Tokenizer:")
    print("max word count: " + str(max_word_count))
    print("Words found: " + str(vocab))
    print(np.array(tokens[::2]).shape)
    print(np.array(tokens[1::2]).shape)

    return (vocab, tokens[::2], tokens[1::2])

def loadTokenizer(path):
    pickle_in = open(path, "rb")
    tokenizer = pickle.load(pickle_in)
    pickle_in.close()
    return tokenizer

def saveTokenizer(path, tokenizer):
    pickel_out = open(path, "wb")
    pickle.dump(tokenizer, pickel_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickel_out.close()

# returns a reply to the given texts
# predicts according to settings
# model: production = 'models/production.h5'
# tokenizer: production = 'tokenizer/p_tokenizer_pi.obj'
def predict_production(text):
    model            = load_model(settings['model']['production'])
    tokenizer        = loadTokenizer(settings['tokenizer']['production'])
    max_word_count   = settings['tokenizer']['max_word_count']
    vocab            = len(tokenizer.word_counts)
    word_sequence    = text_to_word_sequence(text)
    tokens           = tokenizer.texts_to_sequences([word_sequence])[0]
    tokens           = np.array([tokens + [tokenizer.oov_token] * (max_word_count - len(tokens))])#.astype('float32') / vocab
    predicted_tokens = list(filter(lambda x: x != 1, (model.predict(tokens).astype('int')).tolist()[0]))
    return predicted_tokens