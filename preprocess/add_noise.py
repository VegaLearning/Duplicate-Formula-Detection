# add noise in test images

import random
import skimage
from skimage import io
import os
import matplotlib.pyplot as plt

IMAGE_PATH = '../images/imgs/'
LABEL_TEST_PATH = '../images/labels_test.txt'
NOISE_IMAGE_PATH = '../images/imgs_noise'


with open(LABEL_TEST_PATH, 'r', encoding='utf-8') as f_test:
    cnt = 0
    for line in f_test:
        line = line.strip(' '); line = line.strip('\n')
        uuid = line.split('\t')[0]
        # print(uuid)
        img = io.imread(IMAGE_PATH + uuid + '.png')

        # s&p
        amount_noise = 0.01
        img = skimage.util.random_noise(img, mode='s&p', amount=amount_noise)
        # SAVE_PATH = '%s_%s/'%(NOISE_IMAGE_PATH,str(amount_noise))

        # gaussian
        var = 0.001
        img_noise = skimage.util.random_noise(img, mode='gaussian', mean=0, var=var)
        SAVE_PATH = '%s_%s_%s/' % (NOISE_IMAGE_PATH, 'gaussian',str(var))

        if(not os.path.exists(SAVE_PATH)):
            os.mkdir(SAVE_PATH)
        SAVE_PATH_IMG = '%s%s.png'%(SAVE_PATH,uuid)
        io.imsave(SAVE_PATH_IMG,img_noise)
        # plt.imshow(img_noise)
        # plt.show()
        cnt += 1
        print(cnt)
