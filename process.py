import cv2


def process(photos):
    for photo in photos:
        img = cv2.imread(photo)
        cv2.imshow('image', img)
        key = cv2.waitKey(1000)
    pass