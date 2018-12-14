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
IMAGE_PATH = '../images/imgs_noise/imgs_noise_gaussian_s&p_0.005/'
RESULTS_PATH = './test_resnet_logs.txt'
PARAM_SAVE_PATH = './resnet_parameters/'
PRINT_STEP = 10
BATCH_SIZE = 64
Bias_res = - 0.5
SIM_PROP = 0.5 # proportion of sim_pair_data in all test_pair_data

# define model
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.res_layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.res_layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.res_layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        # self.res_layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

        self.avepool =  nn.AvgPool2d(2, 2)

        self.fc = nn.Linear(256, 1)

    def make_layer(self, resblock, outchannel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   # strides = [stride, 1]
        layers = []
        for stride in strides:
            layers.append(resblock(self.inchannel, outchannel, stride))
            self.inchannel = outchannel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_layer1(x)
        x = self.avepool(x)
        x = self.res_layer2(x)
        x = self.avepool(x)
        x = self.res_layer3(x)
        # x = self.res_layer4(x)
        x = F.avg_pool2d(x, (2, 4))
        x = x.view(x.size(0), -1)
        y_pred = self.fc(x)
        return y_pred

def main():
    Model = ResNet()
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
        labels_res = [-1 if ele == 0 else 1 for ele in labels]

        # to_tensor
        imgs = torch.stack(imgs)  # stack image pairs, size = [batch_size,8,64,64]
        labels_res = torch.from_numpy(np.array(labels_res)).long()  # (1 or 0) ,size = [batch,]

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels_res = labels_res.cuda()

        y_pred = Model(imgs).squeeze()
        labels_f = labels_res.float()
        loss = torch.mean(torch.clamp(1 - labels_f * y_pred, min=0.0))

        if step % PRINT_STEP == 0:
            print('%d step, test loss : %.3f' % (step,loss.data.numpy()))
            fw.write('%d step, test loss : %.3f\n' % (step,loss.data.numpy()))

        pred_y = [1 if ele > Bias_res else 0 for ele in y_pred]

        print('dis:', list(y_pred.data.numpy()))
        print('pred_y:',pred_y)
        print('labels:',labels)

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