# util.py
# set of utility functions to make things smoother
# includes: 
# loadSettings, saveTokenizer, loadTokenizer

from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
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
    raw_sequences = [text_to_word_sequence( line ) for line in raw_data]

    tokens        = tokenizer.texts_to_sequences(raw_sequences)    

    # TODO look into a better way of determining max_word_count
    max_word_count = settings['tokenizer']['max_word_count']
    vocab          = len(tokenizer.word_counts)

    # pad to max size
    # todo: possibly set limit for this
    # convert to float and normalize
    #tokens = [sub_tokens + [tokenizer.oov_token] * (max_word_count - len(sub_tokens)) for sub_tokens in tokens]
#
    ## its expected that [::2] is a user question and [1::2] is the bots answer
    ## split array in two for input/anwers
    #input_tokens = tokens[::2]
    #input_set = []
    #for sentence in input_tokens:
    #    for token in sentence:
    #        input_set.append(token)
    #input_set = np.array(input_set)
#
    #ans_tokens = tokens[1::2]
    #ans_set = to_categorical(ans_tokens, vocab + 2)

    print("Tokenizer:")
    print("max word count: " + str(max_word_count))
    print("Words found: " + str(vocab))
    #print(input_set.shape)
    #print(ans_set.shape)


    ## NEW STUFF
    all_tokens = [token for sentence in tokens for token in sentence]

    return (vocab, all_tokens[::2], all_tokens[1::2])
    #return (vocab, input_set, ans_set)

def loadTokenizer(path):
    pickle_in = open(path, "rb")
    tokenizer = pickle.load(pickle_in)
    pickle_in.close()
    return tokenizer

def saveTokenizer(path, tokenizer):
    pickel_out = open(path, "wb")
    pickle.dump(tokenizer, pickel_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickel_out.close()


# converts the token a vector that uniqley identifies the
# token by assigning its index to 1, the rest zero
def token_to_vec(token, tokenizer):
     vocab = len(tokenizer.word_counts) # this will be our vector length
     vec = np.zeros(vocab + 2)
     vec[token] = 1 
     return vec


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


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y