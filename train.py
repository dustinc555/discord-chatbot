#!/usr/bin/python
from keras.models import Sequential
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras import losses
from keras.callbacks import TensorBoard
from keras.models import load_model

import os
import sys

import util


from keras.optimizers import Adam

# ctr-c press event handler
# save model and quit
# this is for when i get tired of waiting
import signal
def signal_handler(sig, frame):
        print('\nsaving')
        model.save(settings['model']['production'])
        print('quiting\n')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

settings = util.loadSettings()

model = load_model(settings['model']['production'])
t = util.loadTokenizer(settings['tokenizer']['production'])

# load training data
# this does refit the tokenizer
(vocab, input_data, context_user, context_bot, next_data_categorical, n) = util.load_and_process_data(settings['files']['training'], t)


num_steps = settings['tokenizer']['max_word_count']
batch_size = settings['training']['batch_size']
num_epochs = settings['training']['epochs']


model.fit([input_data, context_user, context_bot], next_data_categorical,
         batch_size=batch_size,
         epochs=num_epochs)

## Save Model
model.save(settings['model']['production'])
