import numpy as np
import cv2
import os
import argparse
from wagon_counting import gap_cnn as pgc
# from wagon_counting import pretrained_gap_cnn as pgc

model = None
actual_wagon = 'locomotive'
previous_result = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--sequence_path")
    ap.add_argument("-m", "--mode")
    args = vars(ap.parse_args())
    return args


def increment_wagon():
    global actual_wagon
    if actual_wagon == 'locomotive':
        actual_wagon = 1
    else:
        actual_wagon += 1


def get_wagon_number(image):
    global previous_result
    actual_result = contains_gap(image)
    if actual_result and not previous_result:
        increment_wagon()

    previous_result = actual_result
    return actual_wagon


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
            wagon_number = get_wagon_number(img)
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
