import os
import shutil
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import cv2
import numpy as np
import argparse
from random import shuffle
from keras.utils import to_categorical

model_name = 'pre_wagon_gaps_1.h5'


def splitDataset(base_dir, trainCountGap, valCountGap, trainCountOther, valCountOther):
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
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    conv_base.summary()
    conv_base.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def drawPlots(history, handleOverfitting=False):
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
    base_dir = args['dataset_path']
    validation_dir = base_dir + '\\validation'
    train_dir = base_dir + '\\train'

    splitDataset(base_dir, args['train_examples_gap'], args['val_examples_gap'], args['train_examples_other'],
                 args['val_examples_other'])

    handleOverfitting = False

    model = buildNetwork(True)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
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
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

    model.save_weights('models\\' + model_name)

    drawPlots(history, handleOverfitting)

    pass


if __name__ == '__main__':
    main()
