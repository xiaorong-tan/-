from skimage import io, color, feature
import matplotlib.pyplot as plt
import numpy as np
import os

train_dir0 = 'D:/xuelangAI/train_zhengchang/'
train_dir1 = 'D:/xuelangAI/train_feizhengchang/'
files = os.listdir(train_dir1)

import cv2
img1 = cv2.imread(train_dir1+files[0])
sift= cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img1, None)
 
img = cv2.drawKeypoints(img1, keypoints, None)
cv2.imshow("Image", img)
cv2.waitKey(30)
cv2.destroyAllWindows()

