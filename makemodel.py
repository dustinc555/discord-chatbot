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

def maxPool(model, size):
    model.add(Activation('relu'))
    model.add(Dropout(.1))
    model.add(MaxPooling2D(size))

def upSample(model, size):
    model.add(Activation('relu'))
    model.add(Dropout(.1))
    model.add(UpSampling2D(size))

def conv(model, dim, kernel):
    #model.add(Activation('relu'))
    model.add(Dropout(.1))
    model.add(Conv2D(dim, kernel, padding='same', activation='relu'))

## Conv2DTranspose
# essentially an inverse convolution
def convTranspose(model, dim, kernel):
    model.add(Activation('relu'))
    model.add(Dropout(.1))
    model.add(Conv2DTranspose(dim, kernel, padding='same'))

## AtrousConvolution2D
# convolution that isn't necessarily performed on every pixel
def aconv(model, dim, kernel):
    model.add(Activation('relu'))
    model.add(Dropout(.1))
    model.add(AtrousConvolution2D(dim, kernel, padding='same'))

## bottleneck layer
# crunches input down to a (1, 1, n)
# then upsamples back up
# performing convolutions along the way
def bottleneck(model):
    conv(model, 100, (3, 3))
    maxPool(model, (2, 2))

    conv(model, 100, (5, 5))
    maxPool(model, (5, 5))

    conv(model, 100, (5, 5))
    maxPool(model, (5, 5))

    conv(model, 100, (3, 3))
    maxPool(model, (5, 5))

    conv(model, 100, (10, 10))
    upSample(model, (2, 2))

    conv(model, 100, (3, 3))
    upSample(model, (5, 5))

    conv(model, 100, (8, 8))
    upSample(model, (5, 5))

    conv(model, 100, (6, 6))
    upSample(model, (5, 5))

# load input and answer sets
input_images = util.loadImageSet(sys.argv[1]);
ans_images   = util.loadImageSet(sys.argv[2]);



# use 5% of our data as validation data
# validation data is not used in the training but it tells us
# how well our model behaves on something it hasnt seen before
# this is good to help us prevent overfitting
val_percent = .05
validation_input = input_images[int(len(input_images)*(1-val_percent)):]
validation_ans   = ans_images[int(len(ans_images)*(1-val_percent)):]

input_images = input_images[0:int(len(input_images) * (1-val_percent))-1]
ans_images   = ans_images[0:int(len(ans_images) * (1-val_percent))-1]

if (input_images.size != ans_images.size):
    raise ValueError('input set isnt same length as answer set')

## Convert to float and Normalize
input_images = input_images.astype('float32') / 255
ans_images   = ans_images.astype('float32') / 255


model = Sequential()

bottleneck(model)
conv(model, 3, (3, 3))

## Compile Model
model.compile(loss='mse', optimizer='rmsprop')

## Fit Model
# do a single epoch to give it some shape
model.fit(input_images, ans_images,
        batch_size=1,
        epochs=1,
        validation_data=(validation_input, validation_ans),
        shuffle=True)


## save model
model.save(settings.productionModel)
