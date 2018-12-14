# get pair test image uuid
import random
LABEL_TEST_PATH = '../images/labels_test.txt'
PAIR_TEST_PATH = '../images/pairlabes_test_all.txt'
PAIR_TEST_SIM_PATH = '../images/pairlabes_test_sim.txt'
PAIR_TEST_NONSIM_PATH = '../images/pairlabes_test_nonsim.txt'
PAIR_TEST_LOG = '../images/pairlabes_test_log.txt'
uuid2latex = {}
latex2uuid = {}
data_all = []

with open(LABEL_TEST_PATH, 'r', encoding='utf-8') as fp:
    cnt = 0
    for line in fp:
        line = line.strip(' '); line = line.strip('\n')
        uuid = ''; latex = ''; flag = 1
        for ele in line:
            if not flag:
                latex += ele
            if ele == '	':
                flag = 0
            if flag:
                uuid += ele
        data_row = [uuid,latex]
        data_all.append(data_row)
        uuid2latex[uuid] = latex
        if not latex2uuid.get(latex):
            latex2uuid[latex] = set([uuid])
        else:
            latex2uuid[latex].add(uuid)
        # cnt = cnt + 1
        # if cnt >= 100:
        #     break

batch_uuid = []
sim_batch_uuid = []
nonsim_batch_uuid = []
vis_uuid_pair = {}
rd_seed = 2
random.seed(rd_seed)
for epoch in range(10):
    for row in data_all:
        uuid = row[0]
        latex = row[1]
        uuid_pair = uuid + ' ' + uuid
        vis_uuid_pair[uuid_pair] = 1
        set_uuid = latex2uuid[latex]

        # get similar pair
        for i in range(3):
            rand_uuid = random.sample(set_uuid, 1)[0]
            uuid_pair = min(uuid,rand_uuid) + ' ' + max(uuid,rand_uuid)
            if uuid_pair not in vis_uuid_pair:
                sim_uuid = [1, uuid]
                sim_uuid.append(rand_uuid)
                batch_uuid.append(sim_uuid)
                sim_batch_uuid.append(sim_uuid)
                break

        # get non similar pair
        rand_uuid = random.sample(data_all, 1)[0][0]
        uuid_pair = min(uuid, rand_uuid) + ' ' + max(uuid, rand_uuid)
        if (rand_uuid not in set_uuid and uuid_pair not in vis_uuid_pair):
            nonsim_uuid = [0, uuid]  # label 0
            nonsim_uuid.append(rand_uuid)
            batch_uuid.append(nonsim_uuid)
            nonsim_batch_uuid.append(nonsim_uuid)

fw_log = open(PAIR_TEST_LOG,'w',encoding='utf-8')
fw = open(PAIR_TEST_PATH,'w',encoding='utf-8')
fw_sim = open(PAIR_TEST_SIM_PATH,'w',encoding='utf-8')
fw_nonsim = open(PAIR_TEST_NONSIM_PATH,'w',encoding='utf-8')
print('pair size: %d'%len(batch_uuid))
fw_log.write('pair size: %d\n'%len(batch_uuid))

print('simlar pair num : %d'%len(sim_batch_uuid))
fw_log.write('simlar pair num : %d\n'%len(sim_batch_uuid))

print('non-simlar pair num : %d'%len(nonsim_batch_uuid))
fw_log.write('simlar pair num : %d\n'%len(nonsim_batch_uuid))

random.shuffle(batch_uuid)
random.shuffle(sim_batch_uuid)
random.shuffle(nonsim_batch_uuid)

for line in batch_uuid:
    fw.write(str(line)+'\n')

for line in sim_batch_uuid:
    fw_sim.write(str(line)+'\n')

for line in nonsim_batch_uuid:
    fw_nonsim.write(str(line)+'\n')



