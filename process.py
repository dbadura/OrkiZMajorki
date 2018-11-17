import cv2

import OutputCsv
import ocr_test
import readPhotos
from uic import uic_finder
from wagon_counting import gap_detection
import time

import re

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
    outputCsv.frame_number = image_no_extractor.search(image_path)[0]
    return outputCsv


def process(images):
    for image_path in images:
        output = extract_basic_data(image_path)
        img = cv2.imread(image_path)

        start = time.time()

        type = gap_detection.get_wagon_number(image=img)
        if type == 'locomotive':
            output.uic_0_1 = type
        else:
            output.wagon = type

        uic=uic_finder.predict(img)
        if uic!=0: # uic not found
            uic = ocr_test.get_UIC_from_photo(img)

        output.uic_label = uic
        if uic == '0' or uic == '999':
            output.uic_0_1 = 0
        else:
            output.uic_0_1 = 1

        end = time.time() - start
        # print("time: ", end*1000)
        print(output)
        img = cv2.resize(img, (500, 500))
        cv2.imshow('image', img)
        key = cv2.waitKey(1)

    gap_detection.actual_wagon = 'locomotive'
    gap_detection.previous_result = False
    pass


def main():
    readPhotos.startProcessing()


if __name__ == '__main__':
    main()
