# split labels of dataset into train and test
# get 10000 uuid-latex lines from labels randomly as test dataset
import random

LABEL_PATH_RD = '../images/labels_all_replace.txt'
LABEL_TRAIN_PATH = '../images/labels_train.txt'
LABEL_TEST_PATH = '../images/labels_test.txt'


fr = open(LABEL_PATH_RD, 'r', encoding='utf-8')
all_data = fr.read().strip('\n').split('\n')
# print(all_data)

num_line = len(all_data)
print('data_all size:',num_line)

rd_seed = 1
random.seed(rd_seed)
random.shuffle(all_data)

f_train = open(LABEL_TRAIN_PATH, 'w', encoding='utf-8')
f_test = open(LABEL_TEST_PATH, 'w', encoding='utf-8')
for i,line in enumerate(all_data):
    if(i < (num_line-10000)):
        f_train.write(line+'\n')
    else:
        f_test.write(line+'\n')




