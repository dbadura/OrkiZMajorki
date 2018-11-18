import os
import readPhotos
import process
from OutputCsv import Output

train_path = "./data/Validation/0_51/"
right_metadata = "submission_sample.csv"
output_file='./51.csv'

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
        csv.team_name=line[0]
        csv.frame_number=line[3]
        csv.wagon=line[4]
        csv.uic_0_1=line[5]
        csv.uic_label=line[6].rstrip()
        data.append(csv)
    return data

def compare(oryginal, our):
    if len(oryginal)==len(our):
        print ('size is the same')

    for i in range(len(our)):
        print(our[i].compare(oryginal[i]))

    pass

def create_summary_each_dataset(train_path):
    dirs = os.listdir(train_path)
    # for dir in dirs:
    # left_dir = train_path + dir + '/' + dir + '_' + 'left'
    readPhotos.BASE_DIR = train_path
    # process.main()
    oryginal_data=read_data(right_metadata)
    our_data=read_data(output_file)
    compare(oryginal_data, our_data)
    pass



if __name__ == '__main__':
    main()
