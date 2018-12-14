import torch
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import PIL
import numpy as np
import math
import os
from load_data import *
import torch.nn.functional as F

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # select a gpu
IMAGE_PATH = '../images/imgs_noise/imgs_noise_gaussian/'
RESULTS_PATH = './test_logs.txt'
PARAM_SAVE_PATH = './model_parameters/'
PRINT_STEP = 10
Dropout_prob = 0
Bias = 1.3  # y_pred = 1(sim) if eu_dis < Bias else y_pred = 0(non-sim)
BATCH_SIZE = 64
SIM_PROP = 0.5 # proportion of sim_pair_data in all test_pair_data

# define model
class Simese(nn.Module):
    def __init__(self):
        super(Simese, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=1, padding=2),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 16, 5, stride=1, padding=2),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, 5, stride=1, padding=2),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 64, 7, stride=1, padding=0),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True)
        )

        for ele in self.modules():
            if isinstance(ele, nn.Conv2d) or isinstance(ele, nn.Linear):
                nn.init.xavier_uniform(ele.weight)
                # nn.init.kaiming_uniform(ele.weight)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64)
        return x

def main():
    Model = Simese()
    if torch.cuda.is_available():
        Model.cuda()

    # loading model parameters....
    if (os.path.exists(PARAM_SAVE_PATH)):
        print('loading model parameters....')
        paras_load = torch.load(os.path.join(PARAM_SAVE_PATH, 'paras.pkl'),map_location='cpu')
        Model.load_state_dict(paras_load['model'])
        epoch = paras_load['epoch']

    # image transform preprocessing
    img_transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),   # [0,1.0]
        transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5]) # (image-mean)/std = [-1.0,1.0] (R,G,B)
    ])

    # test
    limit_size = min(len(test_sim_data),len(test_nonsim_data))
    if SIM_PROP >= 0.5:
        s_num = limit_size
        test_data_all = test_sim_data[:s_num]
        non_num = int(((1 - SIM_PROP) / SIM_PROP) * limit_size)
        test_data_all.extend(test_nonsim_data[:non_num])
    else:
        non_num = limit_size
        test_data_all = test_nonsim_data[:non_num]
        s_num = int(((SIM_PROP) / (1 - SIM_PROP)) * limit_size)
        test_data_all.extend(test_sim_data[:s_num])

    random.shuffle(test_data_all)
    fw = open(RESULTS_PATH, 'a', encoding='utf-8')
    print('all data_size: %d'%len(test_data_all))
    fw.write('all data_size: %d\n'%len(test_data_all))
    print('similar pair proportion: %.3f'%SIM_PROP)
    fw.write('similar pair proportion: %.3f\n'%SIM_PROP)
    NUM_TEST_DATASET = len(test_data_all)
    TEST_STEPS = math.ceil(NUM_TEST_DATASET / BATCH_SIZE)

    # testing......
    Model.eval()
    print('testing......')
    fw.write('testing......\n')
    TP = TNplusTP = FPplusTP = FNplusTP = 0
    for step in range(TEST_STEPS):
        st = step * BATCH_SIZE
        ed = min(st + BATCH_SIZE, NUM_TEST_DATASET)
        batch_uuid = test_data_all[st:ed] # [[label,uuid1,uuid2],.....]
        imgs1, imgs2, labels = read_images(batch_uuid, img_transform)

        # to_tensor
        imgs1 = torch.stack(imgs1)  # image pairs, size = [batch_size,4,96,96]
        imgs2 = torch.stack(imgs2)  # other image pairs
        labels = torch.from_numpy(np.array(labels)).long() # (1 or 0) ,size = [batch,]

        if torch.cuda.is_available():
            imgs1 = imgs1.cuda()
            imgs2 = imgs2.cuda()
            labels = labels.cuda()

        feature_v1 = Model(imgs1)
        feature_v2 = Model(imgs2)
        # cat_v = torch.cat([feature_v1, feature_v2], -1)

        labels_f = labels.float()
        euclidean_distance = F.pairwise_distance(feature_v1, feature_v2)
        loss_contrastive = torch.mean(labels_f * torch.pow(euclidean_distance, 2) +
                                      (1 - labels_f) * torch.pow(torch.clamp(2 - euclidean_distance, min=0.0), 2))

        if step % PRINT_STEP == 0:
            print('%d step, test loss : %.3f' % (step,loss_contrastive.data.numpy()))
            fw.write('%d step, test loss : %.3f\n' % (step,loss_contrastive.data.numpy()))

        pred_y = [1 if ele < Bias else 0 for ele in euclidean_distance] # get pred_y

        # print('dis:', list(euclidean_distance.data.numpy()))
        # print('pred_y:',pred_y)
        # print('labels:',list(labels.numpy()))

        for i,target in enumerate(labels):
            pred = pred_y[i]
            if pred == target:
                TNplusTP += 1
                if pred == 1:
                    TP += 1
            if pred == 1:
                FPplusTP += 1
            if target == 1:
                FNplusTP += 1

    print('%d step, test loss : %.3f' % (step, loss_contrastive.data.numpy()))
    fw.write('%d step, test loss : %.3f\n' % (step, loss_contrastive.data.numpy()))

    # metric
    if FPplusTP == 0:
        FPplusTP += 1
    if FNplusTP == 0:
        FNplusTP += 1

    Accuracy = TNplusTP / NUM_TEST_DATASET
    Precision = TP / FPplusTP
    Recall = TP / FNplusTP
    F1_score = 2*Precision*Recall/(Precision+Recall)
    print('Accuracy: %.3f' % Accuracy)
    fw.write('Accuracy: %.3f\n' % Accuracy)
    print('Precision: %.3f' % Precision)
    fw.write('Precision: %.3f\n' % Precision)
    print('Recall: %.3f' % Recall)
    fw.write('Recall: %.3f\n' % Recall)
    print('F1_score: %.3f\n' % F1_score)
    fw.write('F1_score: %.3f\n\n' % F1_score)
    epoch += 1

def read_images(batch_uuid, img_transform):
    imgs1 = []  # one image of pair
    imgs2 = []  # anther image of pair
    labels = []  # pair label  (sim = 1, non_sim = 0)
    for pair_uuid in batch_uuid:
        labels.append(pair_uuid[0])
        # shape = (4,96,96)
        img_trans1 = img_transform(PIL.Image.open(IMAGE_PATH + pair_uuid[1] + '.png').convert('RGBA'))
        imgs1.append(img_trans1)
        img_trans2 = img_transform(PIL.Image.open(IMAGE_PATH + pair_uuid[2] + '.png').convert('RGBA'))
        imgs2.append(img_trans2)
    return imgs1,imgs2,labels

if __name__ == "__main__":
    main()