# util.py
# set of utility functions to make things smoother
# includes: 
# loadSettings, saveTokenizer, loadTokenizer

import numpy as np
import os
import cv2
import glob
import toml
import pickle

# returns settings dictionary
# must be in same dir as settings.toml to work
def loadSettings():
	f = open("settings.toml")
	settings_content = f.read()
	f.close()
	return toml.loads(settings_content)

# returns 2 2d numpy arrays
# (input, answer) of tokens
# tokenizer: https://keras.io/preprocessing/text/
# note: a value must be specified for unknown words
# or else it will be ignored and arrays of incorrect size 
# returned
def loadTrainingData(path_to_data, tokenizer):

    raw_text  = open(path_to_data).readlines()
    oov_token = tokenizer.oov_token

    # creates a map for words to tokens
    # higher frequency words will have a smaller
    # token value
    tokenizer.fit_on_text(raw_text)
    print("Tokenizer:")
    print(tokenizer.word_counts)
    print(tokenizer.document_count)
    print(tokenizer.word_index)
    print(tokenizer.word_docs)

    # remove newlines or extra content on sides
    # and tokenize text
    # its expected that [::2] is a user question and [1::2] is the bots answer
    raw_text = [tokenizer.text_to_word_sequence(line.strip()) for line in raw_text]

    # get max length sentence
    max_sentence_length = len(max(raw_text))

    # pad to max size
    # todo: possibly set limit for this
    raw_text = [line + [tokenizer.oov_token] * (max_sentence_length - len(line)) for line in raw_text]

    # split array in two for input/anwers
    # note: arr[start:stop:step]
    return (np.array(raw_text[::2]), np.array(raw_text[1::2]))

def loadTokenizer(path):
    pickle_in = open(path, "rb")
    tokenizer = pickle.load(pickle_in)
    pickle_in.close()
    return tokenizer

def saveTokenizer(path, tokenizer):
    pickel_out = open(path, , "wb")
    picke.dump(tokenizer, pickel_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickel_out.close()