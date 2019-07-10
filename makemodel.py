#!/usr/bin/python3
# makemodel.py
# run this file to create the model and corresponding tokenizer

from keras.models import load_model
from keras.models import Sequential
from keras import backend as K
from keras.layers import Activation, Input, Embedding, TimeDistributed, LSTM, Lambda, Dense, Dropout, concatenate
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
        model.save(settings['model']['production'])
        util.saveTokenizer(settings['tokenizer']['production'], t)
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
(vocab, current_data, previous_data, next_data_categorical) = util.loadTrainingData(settings['files']['training'], t)

hidden_size = vocab // 2
batch_size = settings['training']['batch_size']
num_epochs=settings['training']['epochs']


#train_data_generator = util.KerasBatchGenerator(input_data, ans_data, max_word_count, t, batch_size, vocab)


print("max_word_count: " + str(max_word_count))
print("vocab: " + str(vocab))


# Create model
input_layer       = Input(shape=(max_word_count,), name='input_layer')
context_layer     = Input(shape=(max_word_count,), name='context_layer')

LSTM_encoder = LSTM(hidden_size, return_sequences=True)
LSTM_decoder = LSTM(hidden_size, return_sequences=True)

Shared_Embedding = Embedding(output_dim=hidden_size,
                             input_dim=vocab,
                             input_length=max_word_count)

word_embedding_context = Shared_Embedding(context_layer)
context_embedding = LSTM_encoder(word_embedding_context)

word_embedding_answer = Shared_Embedding(input_layer)
answer_embedding = LSTM_decoder(word_embedding_answer)

merge_layer = concatenate([context_embedding, answer_embedding])

out = Dense(vocab, activation='softmax')(merge_layer)

model = Model(input=[context_layer, input_layer], output = [out])

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([current_data, previous_data], next_data_categorical,
         batch_size=batch_size,
         epochs=num_epochs)

# model.fit_generator(train_data_generator.generate(),
#                     len(input_data)//batch_size,
#                     num_epochs)

# save model
model.save(settings['model']['production'])

# save tokenizer
util.saveTokenizer(settings['tokenizer']['production'], t)
