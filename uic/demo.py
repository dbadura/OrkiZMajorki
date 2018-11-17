import numpy as np
import cv2
import os
import argparse
from uic import uic_cnn as uc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--sequence_path")
    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()

    model = uc.build()
    path = args['sequence_path']
    for i, file in enumerate(sorted(os.listdir(path))):
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))

        label = model.predict(np.expand_dims(np.asarray(img), 0))

        img = cv2.resize(img, (550, 550))
        cv2.imshow('frame', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--sequence_path")
    ap.add_argument("-m", "--mode")
    args = vars(ap.parse_args())
    return args
