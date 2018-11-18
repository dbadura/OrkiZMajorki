import os
import shutil
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import argparse
from random import shuffle
from keras.utils import to_categorical

model_name = 'multiclass_wagon_gaps_1.h5'


def splitDataset(base_dir, trainCountGap, valCountGap, trainCountOther, valCountOther, trainLoco, valLoco):
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

        # Wagon gap
    label = 'lokomotywy'
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
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    print(model.summary())
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


def loadData(base_dir):
    training_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_data = []
    train_labels = []

    val_data = []
    val_labels = []

    labels = ['other', 'wagon_gap', 'lokomotywy']
    for i, label in enumerate(labels):
        path = os.path.join(training_dir, label)
        for file in os.listdir(path):
            file = os.path.join(path, file)
            im = cv2.imread(file)
            im = cv2.resize(im, (150, 150))
            im = im.astype('float32') / 255

            y = np.zeros(len(labels))
            y[i] = 1

            train_labels.append(y)
            train_data.append(im)

        path = os.path.join(validation_dir, label)
        for file in os.listdir(path):
            file = os.path.join(path, file)
            im = cv2.imread(file)
            im = cv2.resize(im, (150, 150))

            y = np.zeros(len(labels))
            y[i] = 1

            val_labels.append(y)
            val_data.append(im)
    return train_data, train_labels, val_data, val_labels


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
    ap.add_argument("-tl", "--train_examples_loco", type=int)
    ap.add_argument("-vl", "--val_examples_loco", type=int)
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
    #              args['val_examples_other'], args['train_examples_loco'],
    #              args['val_examples_loco'])

    handleOverfitting = False
    train_data, train_labels, val_data, val_labels = loadData(base_dir)

    model = buildNetwork(True)

    train_data = np.asarray(train_data)
    val_data = np.asarray(val_data)
    train_labels = np.asarray(train_labels)
    val_labels = np.asarray(val_labels)
    history = model.fit(train_data, train_labels, epochs=15, batch_size=64,
                        validation_data=(val_data, val_labels))

    model.save_weights('models\\' + model_name)

    drawPlots(history, handleOverfitting)

    pass


if __name__ == '__main__':
    main()
