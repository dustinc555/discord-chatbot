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
        model.save(settings.productionModel)
        print('quiting\n')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

settings = util.loadSettings()

model = load_model(settings['model']['production'])
t = util.loadTokenizer(settings['tokenizer']['production'])

# load training data
# this does refit the tokenizer 
(max_word_count, vocab, input_data, ans_data) = util.loadTrainingData(settings['files']['training'], t)

# train
model.fit(input_data, ans_data,
          batch_size=settings['training']['batch_size'],
          epochs=settings['training']['epochs'],
          shuffle=True)
          #callbacks=[TensorBoard(log_dir='logs')])

## Save Model
model.save(settings['model']['production'])

# tell dustin im done
sys.stdout.write('\a')
sys.stdout.flush()
