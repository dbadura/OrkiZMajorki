import os
import re

import process

REGEX = '.jpg'

BASE_DIR = './data/'


def gotodir(path):
    photos = list()
    for dir_name in os.listdir(path):
        full_path =path + os.path.sep + dir_name
        if os.path.isdir(full_path):
            gotodir(full_path)

        isImage = dir_name.endswith('.jpg')
        if isImage:
            photos.append(dir_name)
    if(len(photos) != 0 ):
        full_path_photos =  sorted([path+'/'+photo for photo in photos])
        process.process(full_path_photos)
        pass
    # SEND TO JEWS
    pass

def startProcessing():
    for dir in os.listdir(BASE_DIR):
        gotodir(BASE_DIR + dir)

if __name__ == '__main__':
    startProcessing()
