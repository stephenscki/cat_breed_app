import os, sys

import argparse
import numpy as np
from datetime import datetime
import platform
import glob
import shutil
import tensorflow as tf
import pandas as pd


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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score


BATCH_SIZE = 32
NUM_CLASSES = 19

TRAINTEST_SPLIT = 0.2

def main():
    directory = "breed_images/"
    model = load_model('weight_files/20200201-15:00/w-48-0.74.hdf5')

    train_datagen = ImageDataGenerator(
        validation_split=TRAINTEST_SPLIT,
        rescale=1./255)

    validation_generator = train_datagen.flow_from_directory(
        directory,
        target_size=(299,299),
        batch_size=BATCH_SIZE,
        shuffle=False, 
        class_mode=None,
        subset='validation')

    validation_generator.reset()

    probabilities = model.predict_generator(validation_generator, verbose=1, steps=721/BATCH_SIZE)

    predicted_class_indices = np.argmax(probabilities, axis=1)

    labels = (validation_generator.class_indices)
    labels = dict((v,k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames = validation_generator.filenames
    results = pd.DataFrame({"Filename":filenames, "Predictions":predictions})
    results.to_pickle("predictions.pkl")

    #y_pred = np.rint(predictions)
    y_true = validation_generator.classes
    y_true_string = [labels[k] for k in y_true]


    print(multilabel_confusion_matrix(y_true_string, predictions))
    print(f1_score(y_true=y_true_string, y_pred=predictions, average='weighted'))

    #print(results)

if __name__ == '__main__':
	main()
