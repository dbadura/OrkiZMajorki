import os

import cv2

import OutputCsv
import ocr_test
import readPhotos
import rename_photos
from uic import uic_finder
from wagon_counting import gap_detection
import time
import os
import re

BASE_DIR = './data/Validation/0_64/'

image_no_extractor = re.compile(r'(\d{3})')
train_no_extractor = re.compile(r'_(\d*)_')


def extract_basic_data(image_path):
    if 'left' in image_path:
        left_right = 'left'
    else:
        left_right = 'right'
    train_number = train_no_extractor.search(image_path)[0]
    train_number = train_number[1:len(train_number) - 1]
    outputCsv = OutputCsv.Output(train_number, left_right=left_right)

    outputCsv.train_number = train_number
    outputCsv.left_right = left_right

    ## ALWAYS REMEMBER TO RENAME PHOTOS BEFORE PROCESSING
    outputCsv.frame_number = image_no_extractor.search(image_path)[0]

    return outputCsv


def process(images, frame_skip=15, path=''):
    """
    Main function responsible for processing set of image
    :param images: List with images' paths from one train ordered by occurrence
    :param frame_skip: Number of frames after which detection of gaps is started
    :param path: Not used
    :return: nothing
    """

    # Get train number from name of image
    train_number = train_no_extractor.search(images[0])[0]
    train_number = train_number[1:len(train_number) - 1]

    if 'left' in images[0]:
        left_right = 'left'
    else:
        left_right = 'right'

    f = open(train_number + '_' + left_right + ".csv", "w")
    f.write('team_name,train_number,left_right,frame_number,wagon,uic_0_1,uic_label\n')

    images_output = list()
    wagons = 0

    # Main loop for image processing
    for image_path in images:
        output = extract_basic_data(image_path)
        img = cv2.imread(image_path)

        start = time.time()

        type = gap_detection.get_wagon_number(image=img, frame_skip=frame_skip)
        if type == 'locomotive':
            output.uic_0_1 = type
        else:
            output.wagon = type

        uic = uic_finder.predict(img)
        if uic != 0:  # uic is found
            uic = ocr_test.get_UIC_from_photo(img)

        if uic == '':
            uic = '0'
        output.uic_label = uic
        # output.uic_0_1 = 0

        if str(uic) == '0' or str(uic) == '999':
            if output.uic_0_1 !='locomotive':
                output.uic_0_1 = 0
        else:
            if output.uic_0_1 == 'locomotive':
                gap_detection.increment_wagon(frame_skip=frame_skip)
                output.wagon = 1
            output.uic_0_1 = 1

        print(str(output))
        f.write(str(output) + '\n')
        images_output.append(output)
        end = time.time() - start
        # print("time: ", end*1000)

        img = cv2.resize(img, (500, 500))
        cv2.imshow('image', img)
        key = cv2.waitKey(1)

    gap_detection.actual_wagon = 'locomotive'
    gap_detection.previous_result = False
    f.close()
    pass


def main():
    rename_photos.rename_photos(BASE_DIR)
    readPhotos.startProcessing()


if __name__ == '__main__':
    main()
