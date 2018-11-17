from os import walk
import cv2
import os
import argparse
import shutil
from random import shuffle


def get_difference(image, background):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(background_gray, image_gray)

    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    cv2.imshow("Mean Thresh", thresh)

    cv2.imshow('background', background)
    cv2.imshow('foreground', image)
    cv2.imshow('diff', diff)
    cv2.waitKey(0)
    pass


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_path")
    ap.add_argument("-t", "--target_path")
    ap.add_argument("-l1", "--label1")
    ap.add_argument("-l2", "--label2")
    args = vars(ap.parse_args())
    return args


def list_dirs(path):
    dirs = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            dirs.append(full_path)
    return dirs


def copy_image_to(image_path, target_path, label):
    target_dir = os.path.join(target_path, label)
    shutil.copy(image_path, target_dir)


def main():
    args = parse_args()

    label2 = args['label2']
    label1 = args['label1']
    target_path = args['target_path']
    path = args['dataset_path']

    label1_path = os.path.join(target_path, label1)
    label2_path = os.path.join(target_path, label2)
    if not os.path.exists(label1_path):
        os.makedirs(label1_path)

    if not os.path.exists(label2_path):
        os.makedirs(label2_path)

    files = []
    for dir_1 in list_dirs(path):
        for dir_2 in list_dirs(dir_1):
            for i, file in enumerate(os.listdir(dir_2)):
                files.append(os.path.join(dir_2, file))

    shuffle(files)
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (420, 420))
        cv2.imshow('image', img)
        key = cv2.waitKey(0)

        print(file)

        if key == ord('1'):
            copy_image_to(file, target_path, label1)
            pass
        elif key == ord('2'):
            copy_image_to(file, target_path, label2)
            pass


if __name__ == '__main__':
    main()
