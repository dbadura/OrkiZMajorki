from os import walk
import numpy as np
import cv2
import os
import argparse
import gap_cnn as gc

model = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--sequence_path")
    args = vars(ap.parse_args())
    return args


def contains_gap(image):
    global model
    result = False

    if model is None:
        model = gc.build()

    image = cv2.resize(image, (150, 150))
    label = model.predict(np.expand_dims(np.asarray(image), 0))
    if label == 1:
        result = True

    return result


def main():
    args = parse_args()

    model = gc.build()
    path = args['sequence_path']
    for i, file in enumerate(sorted(os.listdir(path))):
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))

        label = model.predict(np.expand_dims(np.asarray(img), 0))
        print(label)

        img = cv2.resize(img, (550, 550))
        cv2.imshow('frame', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
