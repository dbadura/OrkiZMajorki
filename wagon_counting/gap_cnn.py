import argparse
import os
import shutil
from random import shuffle

import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

"""
This script is used to train model for wagon gap detection
using training and validation dataset passed in arguments
"""

# Name of generated model
model_name = 'wagon_gaps_8.h5'


def splitDataset(base_dir, trainCountGap, valCountGap, trainCountOther, valCountOther):
    """
    Splits dataset in base_dir into training and validation datasets
    :param trainCountGap: Number of images in training set for wagon_gap class
    :param valCountGap: Number of images in validation set for wagon_gap class
    :param trainCountOther: Number of images in training set for other class
    :param valCountOther: Number of images in validation set for other class
    :return: path to training dir, path to validation dir
    """

    # Make separate directories for train/test/validation
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)

    # Other
    label = 'wagon_gap'
    os.mkdir(os.path.join(train_dir, label))
    os.mkdir(os.path.join(validation_dir, label))

    dir_path = os.path.join(base_dir, label)

    files = []
    for file in os.listdir(dir_path):
        files.append(file)

    shuffle(files)
    for i in range(trainCountGap):
        shutil.copyfile(os.path.join(base_dir, label, files[i]), os.path.join(train_dir, label, files[i]))
    for i in range(valCountGap):
        shutil.copyfile(os.path.join(base_dir, label, files[i]), os.path.join(validation_dir, label, files[i]))

    # Wagon gap
    label = 'other'
    os.mkdir(os.path.join(train_dir, label))
    os.mkdir(os.path.join(validation_dir, label))

    dir_path = os.path.join(base_dir, label)

    files = []
    for file in os.listdir(dir_path):
        files.append(file)

    shuffle(files)
    for i in range(trainCountOther):
        shutil.copyfile(os.path.join(base_dir, label, files[i]), os.path.join(train_dir, label, files[i]))
    for i in range(valCountOther):
        shutil.copyfile(os.path.join(base_dir, label, files[i]), os.path.join(validation_dir, label, files[i]))

    return train_dir, validation_dir


def buildNetwork(handleOverfitting=False):
    """
    Builds model of cnn to wagon gap detection
    :param handleOverfitting: If true adds dropout layer
    :return: Compiled model
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    if handleOverfitting:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    print(model.summary())
    return model


def preprocessImages(train_dir, validation_dir, handleOverfitting=False):
    """
    Preprocesses images and creates generators for training and validation datasets
    :param train_dir: Path to training dataset
    :param validation_dir: Path to validation dataset
    :param handleOverfitting: If true, uses standard data augumentation
    :return: generator for training images, generator for validation images
    """
    if handleOverfitting:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True, )
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    return train_generator, validation_generator


def drawPlots(history, handleOverfitting=False):
    """
    Draw plots with training/validation accuracy/loss
    :param history:
    :param handleOverfitting:
    :return:
    """
    postfix = '_ignoreOverfitting' if not handleOverfitting else '_handleOverfitting'
    postfix = '_' + model_name[:-3] + postfix

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('plots\\acc' + postfix)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('plots\\loss' + postfix)
    plt.show()


def build():
    """
    Build model for gap detection
    :return:
    """
    handleOverfitting = False
    model = buildNetwork(handleOverfitting)
    model.load_weights('models/' + model_name)
    return model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_path")
    ap.add_argument("-tg", "--train_examples_gap", type=int)
    ap.add_argument("-vg", "--val_examples_gap", type=int)
    ap.add_argument("-to", "--train_examples_other", type=int)
    ap.add_argument("-vo", "--val_examples_other", type=int)
    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()
    base_dir = os.getcwd() + args['dataset_path']
    validation_dir = base_dir + 'validation'
    train_dir = base_dir + 'train'

    # splitDataset(base_dir, args['train_examples_gap'], args['val_examples_gap'], args['train_examples_other'],
    #              args['val_examples_other'])

    handleOverfitting = False

    model = buildNetwork(True)
    train_generator, validation_generator = preprocessImages(train_dir, validation_dir, handleOverfitting)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=50)

    model.save_weights('models\\' + model_name)

    drawPlots(history, handleOverfitting)

    pass


if __name__ == '__main__':
    main()
