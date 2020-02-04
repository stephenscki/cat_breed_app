import os, sys

import argparse
import numpy as np
from datetime import datetime
import platform
import glob
import shutil
import tensorflow as tf


# from IPython.display import display
# from PIL import Image
from keras import backend as K
from keras import regularizers
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout, Activation, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

from PIL import Image

BATCH_SIZE = 32
NUM_CLASSES = 19
NUM_EPOCHS = 30
TRAINTEST_SPLIT=0.2


def buildBaseModel():
    '''
    returns base model of a bottom layer trainable Inception v3
    '''
    return InceptionV3(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(shape=(299,299,3)))

def lastLayerInsertion(model, num_classes):
    '''
    adds the last few layers to fully employ transfer learning and returns it
    '''
    x = model.output
    x = AveragePooling2D(pool_size=(8,8))(x)
    x = Flatten()(x)
    predictions = Dense(num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0005), activation='softmax')(x)
    model = Model(input=model.input, output=predictions)
    return model

def getNumOfTrainingTesting(split):
    '''
    input: double showing the split of validation, and thus the ratio between train/validation
    output: number of images for training, testing based on the split
    Using terminal command of "find DIR_NAME -type f | wc -l", got 3790 as total number of images
    '''
    numImages = 3790
    num_testing = int(3790 * split)
    num_training = 3790 - num_testing
    if split == TRAINTEST_SPLIT:
        assert num_testing==758, 'num_testing = {}'.format(num_testing)
    return num_training, num_testing


def getTrainTestGenerators(directory):
    '''
    input: directory where subfolders representing each breed exist
    output: returns training and testing generators used to augment the dataset.
    '''
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        validation_split=TRAINTEST_SPLIT,
        shear_range=0.1,
        rescale=1./255)

    #test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory,
        target_size=(299,299),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training')

    #using the subset parameter in flow_from_dir, we can use the same directory
    #However, the validation generator now also augments the data.
    #See https://github.com/keras-team/keras/issues/5862
    #given time, will do my own train/test split later
    validation_generator = train_datagen.flow_from_directory(
        directory,
        target_size=(299,299),
        batch_size=BATCH_SIZE,
        shuffle=False, 
        class_mode='categorical',
        subset='validation')

    return train_generator, validation_generator

def compileModel(model):
    '''
    compile model
    '''
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['acc'])

    return None

def setupSaving():
    '''
    input: none
    output: checkpoint to be used in callbacks of model fit_generator
    '''

    directoryForWeightFiles = "weight_files/"+datetime.now().strftime("%Y%m%d-%H:%M")+"/"
    os.mkdir(directoryForWeightFiles)
    
    savePath = directoryForWeightFiles+ "w-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        savePath,
        monitor='val_acc',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='max')
    return checkpoint

def fitModel(model, train_gen, test_gen, num_epochs):
    '''
    input: model, training_generator, testing_generator, number of epochs
    output: none
    model is fitted
    '''


    #saving logs for tensorboard data
    logdir = "../logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)    

    num_training, num_testing = getNumOfTrainingTesting(TRAINTEST_SPLIT)

    checkpoint = setupSaving()
    #fit model
    model.fit_generator(
        train_gen,
        validation_data=test_gen,
        steps_per_epoch=(num_training // BATCH_SIZE),
        epochs=num_epochs,
        validation_steps=(num_testing // BATCH_SIZE),
        callbacks=[tensorboard_callback, checkpoint],
        verbose=1)

    return None

def main(args):
    print("started program")
    num_epochs = int(args.nb_epoch)
    num_classes = NUM_CLASSES

    model = lastLayerInsertion(buildBaseModel(), num_classes)

    trainGenerator, testGenerator = getTrainTestGenerators('breed_images/')
    compileModel(model)
    fitModel(model, trainGenerator, testGenerator, num_epochs)
    print("done")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--nb_epoch", "-e", default=NUM_EPOCHS)
    args = args.parse_args()
    main(args)
