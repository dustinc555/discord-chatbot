#!/usr/bin/python3
# makemodel.py
# run this file to create the model and corresponding tokenizer

from keras.models import load_model
from keras.models import Sequential
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras import losses
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

import tensorflow as tf
import os
import sys
import signal
import pdb

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
t = Tokenizer(oov_token=settings['tokenizer']['oov_token'], 
              num_words=settings['tokenizer']['num_words'])
max_word_count = settings['tokenizer']['max_word_count']

# loadTrainingData fits the data to the tokenizer and
# returns the question and answers as numpy array of tokens we well as the
# number of discovered words (vocab)
(vocab, input_data, ans_data) = util.loadTrainingData(settings['files']['training'], t)

hidden_size = 500
num_steps = max_word_count
batch_size = settings['training']['batch_size']
num_epochs=settings['training']['epochs']


train_data_generator = util.KerasBatchGenerator(input_data, num_steps, batch_size, vocab,
                                           skip_step=num_steps)

model = Sequential()

# embedding layer encodes tokens to unique vector representations
model.add(Embedding(vocab, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(TimeDistributed(Dense(vocab)))
model.add(Activation('softmax'))

# Compile & run training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit_generator(train_data_generator.generate(), len(input_data)//(batch_size*num_steps), num_epochs)

# save model
model.save(settings['model']['production'])

# save tokenizer
util.saveTokenizer(settings['tokenizer']['production'], t)
