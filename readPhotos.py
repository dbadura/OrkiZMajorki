import os
import re

import process

REGEX = '.jpg'

base_dir = './data/'


def change_name(file_name, path):
    isImage = file_name.endswith('.jpg')
    if isImage:
        regex = re.compile(r"\d_\d*_((right)|(left))_|.jpg")
        search = regex.search(file_name)
        print('filename', file_name, 'search', search)
        span_index = search.span(0)[1]
        print(search.span(0))
        rest = file_name[span_index:]

        # print(rest)
        # for x in range(7 - len(rest)):
        #     rest = '0' + rest
        # new_file_name = file_name[:span_index] + rest
        # print('new file name ', new_file_name)


def gotodir(path):
    photos = list()
    for dir_name in os.listdir(path):
        full_path =path + '/' + dir_name
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

def main():
    for dir in os.listdir(base_dir):
        gotodir(base_dir + dir)

if __name__ == '__main__':
    main()
