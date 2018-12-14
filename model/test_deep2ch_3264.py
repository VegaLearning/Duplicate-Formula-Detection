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
IMAGE_PATH = '../images/imgs_noise/imgs_noise_gaussian_0.001/'
RESULTS_PATH = './train_deep2ch_3264_logs.txt'
PARAM_SAVE_PATH = './deep2ch_3264_parameters/'
PRINT_STEP = 10
BATCH_SIZE = 64
Bias_deep2ch = -0.99
SIM_PROP = 0.5 # proportion of sim_pair_data in all test_pair_data

# define model
class stack_layer(nn.Module):
    def __init__(self,inchannel, outchannel, k1, k2):
        super(stack_layer, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(k1, k2), stride=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, kernel_size=(k1, k2), stride=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, kernel_size=(k1, k2), stride=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.stack(x)
        return out

class deep_2ch(nn.Module):
    def __init__(self):
        super(deep_2ch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 96, 5, stride=3),
            nn.BatchNorm2d(96),
        )

        self.pooling =  nn.MaxPool2d((1,2), (1,2))

        self.stack_layer1 = stack_layer(96,96,3,3)
        self.stack_layer2 = stack_layer(96,192,2,3)

        self.fc = nn.Linear(192, 1)

        for ele in self.modules():
            if isinstance(ele, nn.Conv2d) or isinstance(ele, nn.Linear):
                nn.init.xavier_uniform_(ele.weight)
                # nn.init.kaiming_uniform_(ele.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.stack_layer1(x)
        x = self.pooling(x)
        x = self.stack_layer2(x)
        x = x.view(x.size(0),-1)
        y_pred = self.fc(x)
        return y_pred

def main():
    Model = deep_2ch()
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
        transforms.Resize((32,64)),
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
        imgs, labels = read_images(batch_uuid, img_transform)
        labels_2ch = [-1 if ele == 0 else 1 for ele in labels]

        # to_tensor
        imgs = torch.stack(imgs)  # stack image pairs, size = [batch_size,8,64,64]
        labels_2ch = torch.from_numpy(np.array(labels_2ch)).long()  # (1 or 0) ,size = [batch,]

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels_2ch = labels_2ch.cuda()

        y_pred = Model(imgs).squeeze()
        labels_f = labels_2ch.float()
        loss = torch.mean(torch.clamp(1 - labels_f * y_pred, min=0.0))

        if step % PRINT_STEP == 0:
            print('%d step, test loss : %.3f' % (step,loss.data.numpy()))
            fw.write('%d step, test loss : %.3f\n' % (step,loss.data.numpy()))

        pred_y = [1 if ele > Bias_deep2ch else 0 for ele in y_pred]

        # print('dis:', list(y_pred.data.numpy()))
        # print('pred_y:',pred_y)
        # print('labels:',labels)

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

    print('%d step, test loss : %.3f' % (step, loss.data.numpy()))
    fw.write('%d step, test loss : %.3f\n' % (step, loss.data.numpy()))

    # metric
    if FPplusTP == 0:
        FPplusTP += 1
    if FNplusTP == 0:
        FNplusTP += 1

    Accuracy = TNplusTP / NUM_TEST_DATASET
    Precision = TP / FPplusTP
    Recall = TP / FNplusTP
    if Precision + Recall == 0:
        F1_score = 0
    else:
        F1_score = 2 * Precision * Recall / (Precision + Recall)
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
    imgs = []  # stack pair image
    labels = []  # pair label  (sim = 1, non_sim = 0)
    for pair_uuid in batch_uuid:
        labels.append(pair_uuid[0])
        # shape = (4,64,64)
        img_trans1 = img_transform(PIL.Image.open(IMAGE_PATH + pair_uuid[1] + '.png').convert('RGBA'))
        img_trans2 = img_transform(PIL.Image.open(IMAGE_PATH + pair_uuid[2] + '.png').convert('RGBA'))
        img_stack = torch.cat([img_trans1,img_trans2],0)
        # print(img_stack.shape)
        imgs.append(img_stack)
    return imgs,labels

if __name__ == "__main__":
    main()