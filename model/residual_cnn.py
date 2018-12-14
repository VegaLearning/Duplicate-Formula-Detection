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
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # select a gpu
IMAGE_PATH = '../images/imgs/'
RESULTS_PATH = './train_resnet_logs.txt'
PARAM_SAVE_PATH = './resnet_parameters/'
PRINT_STEP = 10
EPOCH = 30
LR = 0.0005
Dropout_prob = 0
Bias_res = 0
BATCH_SIZE = 64
LABEL_BATCH = int(BATCH_SIZE/2)

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
        x = F.avg_pool2d(x, (2,4))
        x = x.view(x.size(0), -1)
        y_pred = self.fc(x)
        return y_pred

def main():
    Model = ResNet()
    if torch.cuda.is_available():
        Model.cuda()

    # loading model parameters....
    if (os.path.exists(PARAM_SAVE_PATH+'paras.pkl')):
        print('loading model parameters....')
        paras_load = torch.load(os.path.join(PARAM_SAVE_PATH, 'paras.pkl'))
        Model.load_state_dict(paras_load['model'])
        epoch = paras_load['epoch'] + 1
    else:
        epoch = 1

    Optimizer = torch.optim.Adam(Model.parameters(),lr=LR, betas=(0.9, 0.999))
    # Optimizer = torch.optim.ASGD(Model.parameters(), lr=1e-2, alpha=0.9 ,weight_decay=0.0005)

    # image transform preprocessing
    img_transform = transforms.Compose([
        transforms.Resize((32,64)),
        transforms.ToTensor(),   # [0,1.0]
        transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5]) # (image-mean)/std = [-1.0,1.0] (R,G,B)
    ])

    # train and test
    if is_ploting:
        plt.ion()
        plt.figure(figsize=(10, 6))
    plt_step = []; plt_epoch = []
    plt_loss = []; plt_acc = []; plt_precision = []; plt_recall = []; plt_f1 = [];
    NUM_TRAIN_DATASET = len(train_data_all)
    TRAIN_STEPS = math.ceil(NUM_TRAIN_DATASET/LABEL_BATCH)
    NUM_TEST_DATASET = len(valid_data_all)
    TEST_STEPS = math.ceil(NUM_TEST_DATASET / BATCH_SIZE)
    fw = open(RESULTS_PATH, 'a', encoding='utf-8')
    while epoch <= EPOCH:
        # training......
        Model.train()
        print('In epoch: %d'%(epoch))
        fw.write('In epoch: %d\n'%(epoch))
        print('training......')
        fw.write('training......\n')

        for step in range(TRAIN_STEPS):
            st = step*LABEL_BATCH
            ed = min(st+LABEL_BATCH,NUM_TRAIN_DATASET)
            train_data = train_data_all[st:ed] # [[uuid1,latex1],[uuid2,latex2]....], size = (batch,2)

            # get 2 pair_uuid(similar and non similar) from each row
            batch_uuid = get_pair_uuid(train_data)

            # read image through uuid
            imgs,labels = read_images(batch_uuid, img_transform)
            labels_res = [-1 if ele == 0 else 1 for ele in labels]

            # to_tensor
            imgs = torch.stack(imgs) # stack image pairs, size = [batch_size,8,64,64]
            labels_res = torch.from_numpy(np.array(labels_res)).long() # label pairs(sim = 1, non_sim = 0), size = [batch_size,]

            if torch.cuda.is_available():
                x_var, label_var = Variable(imgs).cuda(), Variable(labels_res).cuda()
            else:
                x_var, label_var = Variable(imgs), Variable(labels_res)

            y_pred = Model(x_var).squeeze()
            labels_f = labels_res.float()
            loss = torch.mean(torch.clamp(1 - labels_f * y_pred, min=0.0))

            if step % PRINT_STEP == 0:
                print('%d step, train loss : %.3f' % (step,loss.data.numpy()))
                fw.write('%d step, train loss : %.3f\n' % (step,loss.data.numpy()))
                if(is_ploting):
                    plt_step.append((epoch-1) * TRAIN_STEPS + step)
                    plt_loss.append(loss.data.numpy())
                    plt.subplot(2, 2, 1)
                    plt.cla()
                    plt.plot(plt_step,plt_loss)
                    plt.xlabel('step'); plt.ylabel('loss')
                    plt.pause(0.1)

            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
        print('%d step, train loss : %.3f' % (step, loss.data.numpy()))
        fw.write('%d step, train loss : %.3f\n' % (step, loss.data.numpy()))
        if is_ploting:
            plt.subplot(2, 2, 1)
            plt.plot(plt_step, plt_loss)
            plt.xlabel('step'); plt.ylabel('loss')
            plt.pause(0.1)

        # saving model parameters....
        print('saving model parameters....')
        if(not os.path.exists(PARAM_SAVE_PATH)):
            os.mkdir(PARAM_SAVE_PATH)
        paras_save = {
            'model':Model.state_dict(),
            'epoch':epoch
        }
        torch.save(paras_save, os.path.join(PARAM_SAVE_PATH,('paras_%d_epoch.pkl'%epoch)))

        # testing......
        Model.eval()
        print('testing......')
        fw.write('testing......\n')
        TP = TNplusTP = FPplusTP = FNplusTP = 0
        for step in range(TEST_STEPS):
            st = step * BATCH_SIZE
            ed = min(st + BATCH_SIZE, NUM_TEST_DATASET)
            batch_uuid = valid_data_all[st:ed] # [[label,uuid1,uuid2],.....]
            imgs, labels = read_images(batch_uuid, img_transform)
            labels_res = [-1 if ele == 0 else 1 for ele in labels]
            # to_tensor
            imgs = torch.stack(imgs) # stack image pairs, size = [batch_size,8,64,64]
            labels_res = torch.from_numpy(np.array(labels_res)).long() # (1 or 0) ,size = [batch,]

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

            # for e1,e2 in zip(y_pred,labels):
            #     print(e1.data,e2)

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
        print('Epoch: %d'%(epoch))
        fw.write('Epoch: %d\n'%(epoch))
        Accuracy = TNplusTP / NUM_TEST_DATASET
        Precision = TP / FPplusTP
        Recall = TP / FNplusTP
        if Precision+Recall == 0:
            F1_score = 0
        else:
            F1_score = 2*Precision*Recall/(Precision+Recall)
        print('Accuracy: %.3f' % Accuracy)
        fw.write('Accuracy: %.3f\n' % Accuracy)

        if is_ploting:
            plt.subplot(2, 2, 2)
            plt.cla()
            plt_epoch.append(epoch)
            plt_acc.append(Accuracy)
            plt.plot(plt_epoch, plt_acc)
            plt.xlabel('epoch'); plt.ylabel('Accuracy')
            plt.pause(0.1)

        print('Precision: %.3f' % Precision)
        fw.write('Precision: %.3f\n' % Precision)
        if is_ploting:
            plt.subplot(2, 3, 4)
            plt.cla()
            plt_precision.append(Precision)
            plt.plot(plt_epoch, plt_precision)
            plt.xlabel('epoch'); plt.ylabel('Precision')
            plt.pause(0.1)

        print('Recall: %.3f' % Recall)
        fw.write('Recall: %.3f\n' % Recall)
        if is_ploting:
            plt.subplot(2, 3, 5)
            plt.cla()
            plt_recall.append(Recall)
            plt.plot(plt_epoch, plt_recall)
            plt.xlabel('epoch'); plt.ylabel('Recall')
            plt.pause(0.1)

        print('F1_score: %.3f\n' % F1_score)
        fw.write('F1_score: %.3f\n\n' % F1_score)
        if is_ploting:
            plt.subplot(2, 3, 6)
            plt.cla()
            plt_f1.append(F1_score)
            plt.plot(plt_epoch, plt_f1)
            plt.xlabel('epoch'); plt.ylabel('F1_score')
            plt.pause(0.1)
            plt.savefig('./train_logs.png')
        epoch += 1
    if is_ploting:
        plt.ioff()
        plt.show

def get_pair_uuid(train_data):
    batch_uuid = []
    for row in train_data:
        uuid = row[0]
        latex = row[1]
        set_uuid = latex2uuid[latex]
        # get similar pair
        sim_uuid = [1, uuid]  # label 1
        if len(set_uuid) == 1:
            sim_uuid.append(uuid)
        else:
            rand_uuid = random.sample(set_uuid, 2)
            if (rand_uuid[0] != uuid):
                sim_uuid.append(rand_uuid[0])
            else:
                sim_uuid.append(rand_uuid[1])
        # print('sim ---',sim_uuid)
        batch_uuid.append(sim_uuid)

        # get non similar pair
        nonsim_uuid = [0, uuid]  # label 0
        while True:
            rand_data = random.sample(train_data_all, 1)
            rand_uuid = rand_data[0]
            if (rand_uuid[0] not in set_uuid):
                nonsim_uuid.append(rand_uuid[0])
                break
                # print('nosim ---',nonsim_uuid)
        batch_uuid.append(nonsim_uuid)
        # print(np.shape(batch_uuid))
    return batch_uuid

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