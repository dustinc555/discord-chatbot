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


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(max_word_count, latent_dim)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,
                           return_state=True)(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


# save model
model.save(settings['model']['production'])

# save tokenizer
util.saveTokenizer(settings['tokenizer']['production'], t)
