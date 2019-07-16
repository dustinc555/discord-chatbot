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
(vocab, input_data, context_user, context_bot, next_data_categorical, next_data) = util.load_and_process_data(settings['files']['training'], t)

hidden_size = 800
batch_size = settings['training']['batch_size']
num_epochs = settings['training']['epochs']


#train_data_generator = util.KerasBatchGenerator(context_questions, context_answers, input_data, next_data_categorical, t, batch_size, vocab)


print("max_word_count: " + str(max_word_count))
print("vocab: " + str(vocab))


# Create model
input_layer = Input(shape=(max_word_count,), name='current_question')
context_layer_user = Input(shape=(max_word_count,), name='context_user')
context_layer_bot = Input(shape=(max_word_count,), name='context_bot')

input_embedding = Embedding(output_dim=hidden_size, input_dim=vocab, input_length=max_word_count)
context_embedding_user = Embedding(output_dim=hidden_size, input_dim=vocab, input_length=max_word_count)
context_embedding_bot = Embedding(output_dim=hidden_size, input_dim=vocab, input_length=max_word_count)

embedding_input = input_embedding(input_layer)
embedding_context_user = context_embedding_user(context_layer_user)
embedding_context_bot = context_embedding_bot(context_layer_bot)

LSTM_encoder = LSTM(hidden_size, return_sequences = True, dropout = .5, recurrent_dropout = .5)
LSTM_decoder = LSTM(hidden_size, return_sequences = True, dropout = .5, recurrent_dropout = .5)

merge_layer = add([embedding_input, embedding_context_user, embedding_context_bot])
encoded_layer = LSTM_encoder(merge_layer)
decoded_layer = LSTM_decoder(encoded_layer)

out = Dense(vocab, activation='softmax')(decoded_layer)

model = Model(input=[input_layer, context_layer_user, context_layer_bot], output = [out])

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([input_data, context_user, context_bot], next_data_categorical,
         batch_size=batch_size,
         epochs=num_epochs)

# model.fit_generator(train_data_generator.generate(),
#                     len(input_data)//batch_size,
#                     num_epochs)


# save model
model.save(settings['model']['production'])

# save tokenizer
util.saveTokenizer(settings['tokenizer']['production'], t)
util.save_training_data([vocab, input_data, context_user, context_bot, next_data_categorical])
