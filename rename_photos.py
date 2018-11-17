import os

REGEX = '.jpg'

base_dir = './data/'
RIGHT_FILE_NAME = 18
LEFT_FILE_NAME = 17


def change_name(file_name, path):
    isImage = file_name.endswith('.jpg')
    if isImage:
        border = 0
        lenght = 0
        if 'left' in file_name:
            border = 10
            lenght = LEFT_FILE_NAME

        elif 'right' in file_name:
            border = 11
            lenght = RIGHT_FILE_NAME

        if (border != 0 and len(file_name) != lenght):
            diff = lenght - len(file_name)
            new_name = file_name
            for i in range(0, diff):
                new_name = new_name[0:border] + '0' + new_name[border:]
            print('old name:' + path + '/' + file_name + ' new name:', path + '/' + new_name)
            os.rename(path + '/' + file_name, path + '/' + new_name)


def gotodir(path):
    for dir_name in os.listdir(path):
        if os.path.isfile(path + '/' + dir_name):
            change_name(dir_name, path)
        else:
            gotodir(path + '/' + dir_name)


for dir in os.listdir(base_dir):
    gotodir(base_dir + dir)
