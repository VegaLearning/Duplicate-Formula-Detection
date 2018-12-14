# delete repetitive and no-image uuid-latex line
import os
LABEL_PATH_RD = '../images/label.split.txt'
LABEL_PATH_WT = '../images/labels_all.txt'
IMAGE_PATH = '../images/imgs/'
debug = False

uuid = {}
img_dict = {}
total_images = os.listdir(IMAGE_PATH)
for ele in total_images:
    one = ele.split('.')
    img_dict[one[0]] = 1
    # print(id[0])
# print(type(total_images))

fw = open(LABEL_PATH_WT, 'w', encoding='utf-8')
with open(LABEL_PATH_RD, 'r', encoding='utf-8') as fp:
    cnt = 0
    for line in fp:
        a = line.split()
        if not uuid.get(a[0]):
            uuid[a[0]] = 1
            if a[0] in img_dict:
                fw.write(line)

        cnt = cnt + 1
        # print(cnt)
        if debug == True:
            if cnt > 5:
                break

