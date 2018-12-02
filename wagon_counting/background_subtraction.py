from os import walk
import numpy as np
import cv2
import os
import argparse

"""
Script using background subtraction between current frame and background taken from first frame

NOTE: Not implemented
"""

def get_difference(image, background):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(background_gray, image_gray)

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])



    eq1 = cv2.equalizeHist(image_gray)
    equalizedImg1 = np.hstack([image_gray, eq1])
    cv2.imshow('equalizedImg1', equalizedImg1)

    eq2 = cv2.equalizeHist(background_gray)
    equalizedImg2 = np.hstack([background_gray, eq2])
    cv2.imshow('equalizedImg2', equalizedImg2)


    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    cv2.imshow("Mean Thresh", thresh)





    cv2.imshow('background', background)
    cv2.imshow('foreground', image)
    cv2.imshow('diff', diff)
    cv2.waitKey(0)
    pass


def equalize(image, background):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    image1 = background_gray * (np.average(background_gray)/np.average(image_gray))
    image2 = image_gray * (np.average(image_gray)/np.average(background_gray))
    image1 = np.asarray(image1, dtype='uint8')
    image2 = np.asarray(image2, dtype='uint8')

    cv2.imshow('background', background_gray)
    cv2.imshow('foreground', image_gray)
    cv2.imshow('background e', image2)
    cv2.imshow('foreground e', image1)
    cv2.waitKey(0)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_path")
    args = vars(ap.parse_args())
    return args


def list_dirs(path):
    dirs = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            dirs.append(full_path)
    return dirs


def main():
    args = parse_args()

    path = args['dataset_path']
    for dir_1 in list_dirs(path):
        for dir_2 in list_dirs(dir_1):
            for i, file in enumerate(os.listdir(dir_2)):
                file_path = os.path.join(dir_2, file)
                img = cv2.imread(file_path)
                if i == 0:
                    background = img

                equalize(img, background)


if __name__ == '__main__':
    main()
