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

    raw_data  = open(path_to_data).readlines()
    max_word_count = settings['tokenizer']['max_word_count']
    oov_token = tokenizer.oov_token

    # convert raw text to arrays of words (sequences)
    raw_sequences = [text_to_word_sequence( line ) for line in raw_data]

    # remove questions and answers that are above the max size
    valid_data = []
    for i in range(0, len(raw_sequences), 2):
        if (len(raw_sequences[i]) <= max_word_count and len(raw_sequences[i + 1]) <= max_word_count):
                valid_data.append(raw_sequences[i])
                valid_data.append(raw_sequences[i + 1])

    # creates a map for words to tokens
    # higher frequency words will have a smaller
    # token value
    # note: ensure char_level is set to false (should be default)
    tokenizer.fit_on_texts(valid_data)

    # convert raw text to arrays of words (sequences)
    tokens = tokenizer.texts_to_sequences(valid_data)
    vocab = len(tokenizer.word_counts) + 2

    # pad to max size
    tokens = [sub_tokens + [tokenizer.oov_token] * (max_word_count - len(sub_tokens)) for sub_tokens in tokens]

    current_data = np.array(tokens[::2])
    next_data = np.array(tokens[1::2])

    previous_data = np.zeros(next_data.shape)
    previous_data[0] = np.full(max_word_count, tokenizer.oov_token)
    for i in range(1, len(current_data)):
            previous_data[i] = next_data[i-1]

    # one hot encoding for answer set
    next_data_categorical = np.array([to_categorical(ans, num_classes=vocab) for ans in next_data])

    return (vocab, current_data, previous_data, next_data_categorical)

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
def token_to_vec(vocab, token, tokenizer):
     vec = np.zeros(vocab)
     vec[token] = 1
     return vec


# returns a reply to the given texts
# predicts according to settings
# model: production = 'models/production.h5'
# tokenizer: production = 'tokenizer/p_tokenizer_pi.obj'
def predict_production(current_text="", previous_text=""):
    model            = load_model(settings['model']['production'])
    tokenizer        = loadTokenizer(settings['tokenizer']['production'])
    max_word_count   = settings['tokenizer']['max_word_count']
    vocab            = len(tokenizer.word_counts)

    word_sequences    = [text_to_word_sequence(current_text), text_to_word_sequence(previous_text)]
    tokens           = tokenizer.texts_to_sequences(word_sequences)
    padded_tokens           = np.array([sub_tokens + [tokenizer.oov_token] * (max_word_count - len(sub_tokens)) for sub_tokens in tokens])


    prediction = model.predict([[padded_tokens[0]],[padded_tokens[1]]])[0]

    predicted_tokens = [np.argmax(tok) for tok in prediction]

    # remove empty space token
    # essentially the oov_token is counted as empty space
    predicted_tokens = list(filter(lambda x: x != tokenizer.oov_token, predicted_tokens))

    # 0 is invalid and will raise an error upon translation
    predicted_tokens = list(filter(lambda x: x != 0, predicted_tokens))

    predicted_text = tokenizer.sequences_to_texts([predicted_tokens])

    return ' '.join(predicted_text)


# used in fitting the model in makemodel.py
# using a generator assists with memory tax
class KerasBatchGenerator(object):

    def __init__(self, data_x, data_y, max_word_count, tokenizer, batch_size, vocab):
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_word_count = max_word_count
        self.tokenizer = tokenizer
        self.current_idx = 0

    def generate(self):
        previous_sentences   = np.zeros(self.max_word_count)
        current_setnences    = np.zeros(self.max_word_count)
        new_sentences        = np.zeros(self.max_word_count)
        while True:
                if self.current_idx >= len(self.data_x):
                        # reset the index back to the start of the data set
                        self.current_idx = 0

                current_setnence = self.data_x[self.current_idx]

                # if we are at 0, there was no previous sentence
                # otherwise set it to last answer
                if self.current_idx == 0:
                        previous_sentences = np.full(self.max_word_count, self.tokenizer.oov_token)
                else:
                        previous_sentences = new_sentences

                new_sentence = self.data_y[self.current_idx]

                self.current_idx += 1
                yield [current_setnences, previous_sentences], to_categorical(new_sentences, num_classes=self.vocab)
