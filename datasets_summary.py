import os
import readPhotos
import process
from OutputCsv import Output

train_path = "./data/Validation/0_51/"
right_metadata = "submission_sample.csv"


output_file = './34_left.csv'
metadata_file = './data/Training/0_34/0_0_34_metadata.txt'


# team_name,train_number,left_right,frame_number,wagon,uic_0_1,uic_label
# 1234,51,left,0,0,locomotive,0

def main():
    create_summary_each_dataset(train_path)


def read_data(file):
    data = list()
    F = open(file, 'r')
    for line in F:
        line = line.split(',')
        csv = Output(line[1], line[2])
        csv.team_name = line[0]
        csv.frame_number = line[3]
        csv.wagon = line[4]
        csv.uic_0_1 = line[5]
        csv.uic_label = line[6].rstrip()
        data.append(csv)
    return data


def compare(oryginal, our):
    if len(oryginal) == len(our):
        print('size is the same')

    for i in range(len(our)):
        if(i<len(oryginal)):
            print(our[i].compare(oryginal[i]))
    print('Koniec')


def compare_with_metadata(data_to_compare):
    F = open(metadata_file, 'r')
    sum=0
    max_wagon=0
    uic_sum=0
    for lines in F:
        print(lines)
        lines=lines.split(',')
        for image_data in data_to_compare:
            max_wagon=image_data.wagon

            print(lines[2].rstrip()[6:])
            if image_data.uic_label==lines[2].rstrip()[6:]:
                uic_sum+=1
        sum+=1

    print('uic_accuracy ', uic_sum/sum)
    print('max wagon', max_wagon, 'wagon in files: ', sum)


def create_summary_each_dataset(train_path):
    dirs = os.listdir(train_path)
    # for dir in dirs:
    # left_dir = train_path + dir + '/' + dir + '_' + 'left'
    readPhotos.BASE_DIR = train_path
    # process.main()
    oryginal_data = read_data(right_metadata)
    our_data = read_data(output_file)
    compare(oryginal_data, our_data)
    compare_with_metadata(our_data)
    pass


if __name__ == '__main__':
    main()
