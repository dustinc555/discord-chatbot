#!/usr/bin/python
from keras.models import Sequential
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import losses
from keras.callbacks import TensorBoard
from keras.models import load_model

import os
import sys
import cv2

import util
import settings

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

# load input and answer sets
input_images = util.loadImageSet(sys.argv[1]);
ans_images   = util.loadImageSet(sys.argv[2]);


val_percent = .05

validation_input = input_images[int(len(input_images)*(1-val_percent)):]
validation_ans   = ans_images[int(len(ans_images)*(1-val_percent)):]

input_images = input_images[0:int(len(input_images) * (1-val_percent))-1]
ans_images   = ans_images[0:int(len(ans_images) * (1-val_percent))-1]

if (input_images.size != ans_images.size):
    raise ValueError('Uhh, input set isnt same length as answer set')

## Convert to float and Normalize
input_images = input_images.astype('float32') / 255
ans_images   = ans_images.astype('float32') / 255


## Load Model
model = load_model(settings.productionModel)

num_epochs  = int(sys.argv[3])
bsize       = int(sys.argv[5])


generate    = sys.argv[4]

# if told to use keras image generator based on input set
if (generate == 'generate'):

        ## train
        # in our case overfitting is exceptionally bad
        # the keras ImageDataGenerator should help out
        datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=10,  # degrees
        width_shift_range=0.1,
        height_shift_range=0.1)

        datagen.fit(input_images)

        model.fit_generator(datagen.flow(input_images, ans_images, batch_size=bsize),
                        steps_per_epoch=len(input_images) / bsize,
                        validation_data=(validation_input, validation_ans),
                        shuffle=True,
                        epochs=num_epochs)
# else train on available images
else:
        model.fit(input_images, ans_images,
                  batch_size=bsize,
                  epochs=num_epochs,
                  validation_data=(validation_input, validation_ans),
                  shuffle=True)
                  #callbacks=[TensorBoard(log_dir='logs')])

## Save Model
model.save(settings.productionModel)

# tell dustin im done
sys.stdout.write('\a')
sys.stdout.flush()
