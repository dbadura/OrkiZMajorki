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
from random import shuffle

original_dataset_dir = cwd = os.getcwd() + '\\data\\train'
base_dir = os.getcwd() + '\\data\\small'

_validation_dir = os.getcwd() + '\\data\\small\\validation'
_train_dir = os.getcwd() + '\\data\\small\\train'
_test_dir = os.getcwd() + '\\data\\small\\test'


def splitDataset(trainCount=200, valCount=50, testCount=50):
    # Make separate directories for train/test/validation
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    # Make separate directories for categories
    labels = ['other', 'wagon_gap']
    for label in labels:
        os.mkdir(os.path.join(train_dir, label))
        os.mkdir(os.path.join(validation_dir, label))
        os.mkdir(os.path.join(test_dir, label))

        dir_path = os.path.join(base_dir, label)

        files = []
        for file in os.listdir(dir_path):
            files.append(file)

        shuffle(files)
        for i in range(trainCount):
            shutil.copyfile(os.path.join(base_dir, label, files[i]), os.path.join(train_dir, label, files[i]))
        for i in range(testCount):
            shutil.copyfile(os.path.join(base_dir, label, files[i]), os.path.join(test_dir, label, files[i]))
        for i in range(valCount):
            shutil.copyfile(os.path.join(base_dir, label, files[i]), os.path.join(validation_dir, label, files[i]))

    return train_dir, test_dir, validation_dir


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


def showAugumentationExamples():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    fnames = [os.path.join(train_dir + "\\cats", fname) for
              fname in os.listdir(train_dir + "\\cats")]
    img_path = fnames[3]
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break
    plt.show()


def loadTestData(path):
    labels = []
    images = []

    dirs = ['wagon_gap', 'other']
    for i, label in enumerate(dirs):
        dir_path = os.path.join(path, label)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (150, 150))
            images.append(img)
            labels.append(float(i))

    return images, labels


def build():
    handleOverfitting = False
    model = buildNetwork(handleOverfitting)
    model.load_weights('wagon_gaps.h5')
    return model


def main():
    test_images, test_labels = loadTestData(os.getcwd() + '\\data\\small\\test')
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)

    handleOverfitting = False
    model = buildNetwork(handleOverfitting)
    train_generator, validation_generator = preprocessImages(_train_dir, _validation_dir, handleOverfitting)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=50)

    model.save_weights('wagon_gaps.h5')

    # a1 = cv2.imread('D:\\Dataset\\train_separator\\test\\wagon_gap\\0_57_right_254.jpg')
    # a2 = cv2.imread('D:\\Dataset\\train_separator\\test\\other\\0_57_right_98.jpg')
    # a1 = cv2.resize(a1, (150, 150))
    # a2 = cv2.resize(a2, (150, 150))
    # a1 = np.expand_dims(a1, 0)
    # a2 = np.expand_dims(a2, 0)
    #
    # v1 = model.predict(np.asarray(a1))
    # v2 = model.predict(np.asarray(a2))

    drawPlots(history, handleOverfitting)
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    pass


if __name__ == '__main__':
    main()
