import cv2

import OutputCsv
import readPhotos
from wagon_counting import gap_detection


def extract_basic_data(image_path):
    if 'left' in image_path:
        left_right = 'left'
    else:
        left_right = 'right'
    train_number = image_path[:2]

    outputCsv = OutputCsv.Output(train_number, left_right=left_right)
    return outputCsv


def process(images, frame_skip):
    for image_path in images:
        output = extract_basic_data(image_path)
        img = cv2.imread(image_path)

        output.train_number = gap_detection.get_wagon_number(frame_skip=frame_skip, image=img)
        img = cv2.resize(img, (250, 250))
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        print(output)
    pass


def main():
    readPhotos.startProcessing()


if __name__ == '__main__':
    main()
