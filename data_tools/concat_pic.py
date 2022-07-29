import os

import cv2
from tqdm import tqdm
import numpy as np
blurred_path = '/home/pc/Hand_select/b_tes'
unblurred_path = '/home/pc/Hand_select/unblurred_1000/'
dst_path = '/home/pc/Desktop/experiments_res/0729'

imgs = os.listdir(blurred_path)
for img in tqdm(imgs):
    if img.split('.')[-1] == 'png':
        img_blurred = cv2.imread(os.path.join(blurred_path,img))
        # for j in range(0,20):
        # img_blurred = cv2.imread(os.path.join(blurred_path,img.split('.')[0]+'.jpg'))
        img_unblurred = np.zeros(img_blurred.shape, np.uint8)
        img_unblurred.fill(200)
        img_con = cv2.hconcat([img_blurred,img_unblurred])
        cv2.imwrite(os.path.join(dst_path,img.split('.')[0]+'.jpg'),img_con,[cv2.IMWRITE_PNG_COMPRESSION, 0])