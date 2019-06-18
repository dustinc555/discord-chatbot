#!/usr/bin/python
# makemodel.py
# run this file to create the model and corresponding tokenizer


from keras.models import Sequential
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras import losses
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

import os
import sys
import signal

import util

from keras.optimizers import Adam

settings = util.loadSettings()


# ctr-c press event handler
# save model and quit
# this is for when i get tired of waiting
def signal_handler(sig, frame):
        print('\nsaving')
        model.save(settings.productionModel)
        print('quiting\n')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# make tokenizer
t = Tokenizer(oov_token=1)

# loadTrainingData fits the data to the tokenizer and
# returns the question and answers as numpy array of tokens
(input_data, ans_data) = util.loadTrainingData(settings['files']['training'], t)

# make model
model = Sequential()
model.add(Dense(20, activation='relu'))
model.add(Dense(max_sentence_length, activation='relu'))

## Compile Model
model.compile(loss='mse', optimizer='rmsprop')

## Fit Model
model.fit(input_data, ans_data,
        batch_size=1,
        epochs=1,
        shuffle=True)


# save model
model.save(settings['model']['production'])

# save tokenizer
util.saveTokenizer(settings['tokenizer']['path'], t)