from PIL import Image
import os
import imghdr

IMAGE_PATH = '../images/imgs/'

total_images = os.listdir(IMAGE_PATH)
cnt = 0
for ele in total_images:
    try:
        Image.open(os.path.join(IMAGE_PATH,ele))
    except OSError:
        os.remove(os.path.join(IMAGE_PATH,ele))
    cnt += 1
    print(cnt)