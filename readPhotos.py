import os

import process

REGEX = '.jpg'


def gotodir(path):
    photos = list()
    if (os.path.isdir(path)):
        for dir_name in os.listdir(path):
            full_path = path + os.path.sep + dir_name
            if os.path.isdir(full_path):
                gotodir(full_path)

            isImage = dir_name.endswith('.jpg')
            if isImage:
                photos.append(dir_name)
        if (len(photos) != 0):
            full_path_photos = sorted([path + os.path.sep + photo for photo in photos])
            process.process(full_path_photos, frame_skip=15, path=path)
            pass
        pass


def startProcessing():
    for dir in os.listdir(process.BASE_DIR):
        gotodir(process.BASE_DIR + dir)


if __name__ == '__main__':
    startProcessing()
