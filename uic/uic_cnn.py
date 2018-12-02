import argparse
import os
import shutil
from random import shuffle

import cv2
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


"""
This script is used to train model for wagons with UIC detection
using training and validation dataset passed in arguments
"""

model_name = 'uic_10.h5'


def splitDataset(base_dir, trainCountGap, valCountGap, trainCountOther, valCountOther):
    # Make separate directories for train/test/validation
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)

    # Other
    label = 'uic'
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
    label = 'not_uic'
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


def loadTestData(path):
    labels = []
    images = []

    dirs = ['uic', 'not_uic']
    for i, label in enumerate(dirs):
        dir_path = os.path.join(path, label)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (150, 150))
            images.append(img)
            labels.append(i)

    return images, labels


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
        epochs=20,
        validation_data=validation_generator,
        validation_steps=50)

    model.save_weights('models\\' + model_name)

    drawPlots(history, handleOverfitting)

    pass


if __name__ == '__main__':
    main()
