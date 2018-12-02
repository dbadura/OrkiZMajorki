import numpy as np
import cv2
import os
import argparse
from wagon_counting import gap_cnn as pgc

"""
This script is used for counting wagons
by using wagon gap detection model and algortihms
for incrementing wagon number
"""

model = None
actual_wagon = 'locomotive'
previous_result = False
counter = 0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--sequence_path")
    ap.add_argument("-f", "--frame_skip", type=int)
    ap.add_argument("-m", "--mode")
    args = vars(ap.parse_args())
    return args


def increment_wagon(frame_skip):
    """
    Increment current wagon label
    :param frame_skip: amount of initial frames to be skipped
    """
    global actual_wagon
    global counter

    counter += 1
    if counter < frame_skip and actual_wagon == 'locomotive':
        actual_wagon = 1
    else:
        actual_wagon += 1


def get_wagon_number(image, frame_skip):
    """
    Returns wagon label (wagon number or 'locomotive') for current frame
    :param image: Frame with wagon
    :param frame_skip: Number to initial frames to skip
    :return: Wagon label
    """

    global previous_result
    actual_result = contains_gap(image)
    if actual_result and not previous_result:
        increment_wagon(frame_skip)

    previous_result = actual_result
    return actual_wagon


def contains_gap(image):
    """
    Calls gap detection model, returns true if gap is detected
    :param image: Frame with wagon
    """
    global model
    result = False

    if model is None:
        model = pgc.build()

    image = cv2.resize(image, (150, 150))
    label = model.predict(np.expand_dims(np.asarray(image), 0))
    if label == 1:
        result = True

    return result


def main():
    args = parse_args()

    model = pgc.build()
    path = args['sequence_path']
    for i, file in enumerate(sorted(os.listdir(path))):
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))

        if args['mode'] == 'gap':
            label = model.predict(np.expand_dims(np.asarray(img), 0))
            # if round(label) < 1:
            #     label = 'Wagon'
            # else:
            #     label = 'Gap'
            print(round(label[0][0]))

        if args['mode'] == 'wagon_number':
            wagon_number = get_wagon_number(img, frame_skip=15)
            print(wagon_number)

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
