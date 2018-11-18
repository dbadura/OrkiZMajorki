import os
import re

REGEX = '.jpg'

base_dir = './data/'


def change_name(file_name, path):
    isImage = file_name.endswith('.jpg')
    if isImage:
        regex = re.compile(r'\d_\d*_((right)|(left))_')
        search = regex.search(file_name)
        print('filename', file_name, 'search', search)
        span_index = search.span(0)[1]
        rest = file_name[span_index:]
        print(rest)
        for x in range(7 - len(rest)):
            rest = '0' + rest
        new_file_name = file_name[:span_index] + rest
        print('new file name ', new_file_name)
        os.rename(path + os.path.sep + file_name, path + os.path.sep + new_file_name)


def gotodir(path):
    if os.path.isdir(path):
        for dir_name in os.listdir(path):
            if os.path.isfile(path + os.path.sep + dir_name):
                change_name(dir_name, path)
            else:
                gotodir(path + os.path.sep + dir_name)


for dir in os.listdir(base_dir):
    gotodir(base_dir + dir)
