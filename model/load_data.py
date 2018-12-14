import random

LABEL_PATH = '../images/labels_train.txt'
PAIR_TEST_PATH = '../images/pairlabes_test_all.txt'
PAIR_TEST_SIM_PATH = '../images/pairlabes_test_sim.txt'
PAIR_TEST_NONSIM_PATH = '../images/pairlabes_test_nonsim.txt'

rd_seed = 3
random.seed(rd_seed)
is_training = False
debug = True
is_ploting = False
num_data_debug= 1000

uuid2latex = {}
latex2uuid = {}
train_data_all = [] # uuid-latex
valid_data_all = [] # pair label(1 or 0)-uuid1-uuid2
test_sim_data = [] # pair label(1)-uuid1-uuid2
test_nonsim_data = [] # pair label(0)-uuid1-uuid2

# get train_data_all
cnt = 0
if is_training:
    with open(LABEL_PATH, 'r', encoding='utf-8') as f_train:
        for line in f_train:
            line = line.strip(' '); line = line.strip('\n')
            uuid = ''; latex = ''; flag = 1
            for ele in line:
                if not flag:
                    latex += ele
                if ele == '\t':
                    flag = 0
                if flag:
                    uuid += ele
            data_row = [uuid,latex]
            train_data_all.append(data_row)
            # uuid_all.append(uuid)
            # latex_all.append(latex)
            uuid2latex[uuid] = latex
            if not latex2uuid.get(latex):
                # print('----',list(latex))
                latex2uuid[latex] = set([uuid])
            else:
                latex2uuid[latex].add(uuid)
            # print(latex2uuid[latex])

            if debug == True:
                cnt = cnt + 1
                if cnt >= num_data_debug:
                    break

# get validation_data_all
if is_training:
    cnt = 0
    with open(PAIR_TEST_PATH, 'r', encoding='utf-8') as f_valid:
        cnt = 0
        for line in f_valid:
            line = line.strip('\n')
            pair_uuid = eval(line)
            valid_data_all.append(pair_uuid)

            if debug == True:
                cnt = cnt + 1
                if cnt >= num_data_debug:
                    break


# get test_data
if not is_training:
    cnt = 0
    with open(PAIR_TEST_SIM_PATH, 'r', encoding='utf-8') as f_test:
        cnt = 0
        for line in f_test:
            line = line.strip('\n')
            pair_uuid = eval(line)
            test_sim_data.append(pair_uuid)

            if debug == True:
                cnt = cnt + 1
                if cnt >= num_data_debug:
                    break

    with open(PAIR_TEST_NONSIM_PATH, 'r', encoding='utf-8') as f_test:
        cnt = 0
        for line in f_test:
            line = line.strip('\n')
            pair_uuid = eval(line)
            test_nonsim_data.append(pair_uuid)

            if debug == True:
                cnt = cnt + 1
                if cnt >= num_data_debug:
                    break

