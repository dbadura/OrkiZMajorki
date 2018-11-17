import os

import process

REGEX = '.jpg'

BASE_DIR = './data/Training/0_2/'


def gotodir(path):
    photos = list()
    if(os.path.isdir(path)):
        for dir_name in os.listdir(path):
            full_path = path + os.path.sep + dir_name
            if os.path.isdir(full_path):
                gotodir(full_path)

            isImage = dir_name.endswith('.jpg')
            if isImage:
                photos.append(dir_name)
        if (len(photos) != 0):
            full_path_photos = sorted([path + os.path.sep + photo for photo in photos])
            process.process(full_path_photos)
            pass
        pass


def startProcessing():
    for dir in os.listdir(BASE_DIR):
        gotodir(BASE_DIR + dir)


if __name__ == '__main__':
    startProcessing()
