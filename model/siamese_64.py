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
RESULTS_PATH = './train_64_logs.txt'
PARAM_SAVE_PATH = './model_64_parameters/'
PRINT_STEP = 10
EPOCH = 10
LR = 0.0005
Dropout_prob = 0
Bias = 1.0  # y_pred = 1(sim) if eu_dis < Bias else y_pred = 0(non-sim)
BATCH_SIZE = 64
LABEL_BATCH = int(BATCH_SIZE/2)

# define model
class Siamese_64(nn.Module):
    def __init__(self):
        super(Siamese_64, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=1, padding=2),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 5, stride=1),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, 5, stride=1, padding=2),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, 5, stride=1, padding=2),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 128, 7, stride=1, padding=0),
            nn.Dropout(Dropout_prob),
            nn.ReLU(inplace=True)
        )

        for ele in self.modules():
            if isinstance(ele, nn.Conv2d) or isinstance(ele, nn.Linear):
                nn.init.xavier_uniform(ele.weight)
                # nn.init.kaiming_uniform(ele.weight)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0),-1) # [batch_size,128]
        return x

def main():
    Model = Siamese_64()
    if torch.cuda.is_available():
        Model.cuda()

    # loading model parameters....
    if (os.path.exists(PARAM_SAVE_PATH+'paras.pkl')):
        print('loading model parameters....')
        paras_load = torch.load(os.path.join(PARAM_SAVE_PATH, 'paras.pkl'),map_location='cpu')
        Model.load_state_dict(paras_load['model'])
        epoch = paras_load['epoch'] + 1
    else:
        epoch = 1

    Optimizer = torch.optim.Adam(Model.parameters(),lr=LR, betas=(0.9, 0.999))

    # image transform preprocessing
    img_transform = transforms.Compose([
        transforms.Resize((64,64)),
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
            imgs1,imgs2,labels = read_images(batch_uuid, img_transform)

            # to_tensor
            imgs1 = torch.stack(imgs1) # image pairs, size = [batch_size,4,28,28]
            imgs2 = torch.stack(imgs2) # other image pairs
            labels = torch.from_numpy(np.array(labels)).long() # label pairs(sim = 1, non_sim = 0), size = [batch_size,]

            if torch.cuda.is_available():
                x_var1, x_var2, label_var = Variable(imgs1).cuda(), Variable(imgs2).cuda(), Variable(labels).cuda()
            else:
                x_var1, x_var2, label_var = Variable(imgs1), Variable(imgs2), Variable(labels)

            feature_v1 = Model(x_var1)
            feature_v2 = Model(x_var2)

            labels_f = labels.float()
            euclidean_distance = F.pairwise_distance(feature_v1, feature_v2)
            loss_contrastive = torch.mean(labels_f * torch.pow(euclidean_distance, 2) +
                                          (1 - labels_f) * torch.pow(torch.clamp(2 - euclidean_distance, min=0.0), 2))

            # print('dis:',euclidean_distance.data.numpy())
            # print('labels:',list(labels.numpy()))

            if step % PRINT_STEP == 0:
                print('%d step, train loss : %.3f' % (step,loss_contrastive.data.numpy()))
                fw.write('%d step, train loss : %.3f\n' % (step,loss_contrastive.data.numpy()))
                if(is_ploting):
                    plt_step.append((epoch-1) * TRAIN_STEPS + step)
                    plt_loss.append(loss_contrastive.data.numpy())
                    plt.subplot(2, 2, 1)
                    plt.cla()
                    plt.plot(plt_step,plt_loss)
                    plt.xlabel('step'); plt.ylabel('loss')
                    plt.pause(0.1)

            Optimizer.zero_grad()
            loss_contrastive.backward()
            Optimizer.step()
        print('%d step, train loss : %.3f' % (step, loss_contrastive.data.numpy()))
        fw.write('%d step, train loss : %.3f\n' % (step, loss_contrastive.data.numpy()))
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
            imgs1, imgs2, labels = read_images(batch_uuid, img_transform)

            # to_tensor
            imgs1 = torch.stack(imgs1) # image pairs, size = [batch_size,4,96,96]
            imgs2 = torch.stack(imgs2) # other image pairs
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

            # print('dis:', euclidean_distance.data.numpy())
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
        print('Epoch: %d'%(epoch))
        fw.write('Epoch: %d\n'%(epoch))
        Accuracy = TNplusTP / NUM_TEST_DATASET
        Precision = TP / FPplusTP
        Recall = TP / FNplusTP
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
    imgs1 = []  # one image of pair
    imgs2 = []  # anther image of pair
    labels = []  # pair label  (sim = 1, non_sim = 0)
    for pair_uuid in batch_uuid:
        labels.append(pair_uuid[0])
        # shape = (4,28,28)
        img_trans1 = img_transform(PIL.Image.open(IMAGE_PATH + pair_uuid[1] + '.png').convert('RGBA'))
        imgs1.append(img_trans1)
        img_trans2 = img_transform(PIL.Image.open(IMAGE_PATH + pair_uuid[2] + '.png').convert('RGBA'))
        imgs2.append(img_trans2)
    return imgs1,imgs2,labels

if __name__ == "__main__":
    main()