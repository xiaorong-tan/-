import os
import  xml.dom.minidom
file_dir = 'D:/xuelangAI/train_feizhengchang/'
files1 = os.listdir(file_dir)
xmls = []
for i in files1:
    if os.path.splitext(i)[1] == '.xml':
        xmls.append(i)

from PIL import Image
import matplotlib.pyplot as plt

# 裁剪瑕疵
threshold0 = 5
threshold1 = 5
for i in range(0,len(xmls)):
    dom = xml.dom.minidom.parse(file_dir+xmls[i])
    root = dom.documentElement
    filename = root.getElementsByTagName('filename')
    imgName = filename[0].firstChild.data
    xmin = root.getElementsByTagName('xmin')
    ymin = root.getElementsByTagName('ymin')
    xmax = root.getElementsByTagName('xmax')
    ymax = root.getElementsByTagName('ymax')
    x_min = int(xmin[0].firstChild.data)
    y_min = int(ymin[0].firstChild.data)
    x_max = int(xmax[0].firstChild.data)
    y_max = int(ymax[0].firstChild.data)
    im = Image.open(file_dir+imgName)
    if x_max+threshold1>2560 and y_max+threshold1<1920:
        if x_min-threshold0>0 and y_min-threshold0>0:
            region = im.crop((x_min-threshold0, y_min-threshold0, x_max, y_max+threshold1))
        if x_min-threshold0>0 and y_min-threshold0<0:
            region = im.crop((x_min-threshold0, y_min, x_max, y_max+threshold1))
        if x_min-threshold0<0 and y_min-threshold0>0:
            region = im.crop((x_min, y_min-threshold0, x_max, y_max+threshold1))
        else:
            region = im.crop((x_min, y_min, x_max, y_max+threshold1))
    elif x_max+threshold1<2560 and y_max+threshold1<1920:
        if x_min-threshold0>0 and y_min-threshold0>0:
            region = im.crop((x_min-threshold0, y_min-threshold0, x_max+threshold1, y_max+threshold1))
        if x_min-threshold0>0 and y_min-threshold0<0:
            region = im.crop((x_min-threshold0, y_min, x_max+threshold1, y_max+threshold1))
        if x_min-threshold0<0 and y_min-threshold0>0:
            region = im.crop((x_min, y_min-threshold0, x_max+threshold1, y_max+threshold1))
        else:
            region = im.crop((x_min, y_min, x_max+threshold1, y_max+threshold1))
    elif x_max+threshold1<2560 and y_max+threshold1>1920:
        if x_min-threshold0>0 and y_min-threshold0>0:
            region = im.crop((x_min-threshold0, y_min-threshold0, x_max+threshold1, y_max))
        if x_min-threshold0>0 and y_min-threshold0<0:
            region = im.crop((x_min-threshold0, y_min, x_max+threshold1, y_max))
        if x_min-threshold0<0 and y_min-threshold0>0:
            region = im.crop((x_min, y_min-threshold0, x_max+threshold1, y_max))
        else:
            region = im.crop((x_min, y_min, x_max+threshold1, y_max))
       
    else:
        if x_min-threshold0>0 and y_min-threshold0>0:
            region = im.crop((x_min-threshold0, y_min-threshold0, x_max, y_max))
        if x_min-threshold0>0 and y_min-threshold0<0:
            region = im.crop((x_min-threshold0, y_min, x_max, y_max))
        if x_min-threshold0<0 and y_min-threshold0>0:
            region = im.crop((x_min, y_min-threshold0, x_max, y_max))
        else:
            region = im.crop((x_min, y_min, x_max, y_max))
        
    region.save("D:/xuelangAI/train_crop_feizhengchang3/"+imgName)
    print('Processed No.%d image'%i)
    

# 分割正常图片
file_dir = 'D:/xuelangAI/train_正常/'
files = os.listdir(file_dir2)
for i in range(0,len(files)):
    im = Image.open(file_dir2+files[i])
    region1 = im.crop((0,0,1280,960))
    region2 = im.crop((1280,0,2560,960))
    region3 = im.crop((0,960,1280,1920))
    region4 = im.crop((1280,960,2560,1920))
    box = [region1, region2, region3, region4]
    for j in range(0,4):
        name = os.path.splitext(files[i])
        box[j].save(('D:/xuelangAI/train_crop正常/'+name[0]+'-%d'+name[1])%j)
    print('Cropping No.%d image'%i)



# 分割训练图片（resize 896*896, 再分割为16张224*224）
file_dir = 'D:/xuelangAI/train_resize896/'
files = os.listdir(file_dir)
height = 224

for i in range(0,len(files)):
    im = Image.open(file_dir+files[i])
    box = []
    for m in range(1,5):
        for n in range(1,5):
            region = im.crop((height*(m-1),height*(n-1),height*m,height*n))
            box.append(region)
    for j in range(0,len(box)):
        name = os.path.splitext(files[i])
        box[j].save(('D:/xuelangAI/train_zhengchang_crop224/'+name[0]+'-%d'+name[1])%j)
    print('Cropping No.%d image'%i)




#height=64
#Region=[]
#pic = [[0 for i in range(16)] for i in range(len(files3))]
#for i in range(0,len(files3)):
#    im = Image.open(dir3+files3[i])
#    for m in range(1,5):#divided into 16 pictures
#        for n in range(1,5):
#             Region.append(im.crop((height*(m-1),height*(n-1),height*m,height*n)))
#    for k in range(0,16):
#        pic[i][k]=np.array(Region[k])#pic.shape=(64, 64, 3)
#Pic=np.array(pic)#Pic.shape=(662, 16)
 
#prob=[]
#probmax=[]
#for i in range(0,662):
#    for j in range(0,16):
#        #rePic = tf.reshape(pic[i][j], [-1,64,64,3])
#        #prob.append(softmax_accuracy(rePic[i][j]))
#        prob.append(softmax_accuracy(pic[i][j]))
#    prob=np.array(prob)
#    probmax.append(np.amax([x[0] for x in prob]  ))  
#    prob=prob.tolist()
#    prob=[]
#probMax=np.array(probmax)
#print(probMax)

# 对非正常图片进行翻转
from PIL import Image
import os
file_dir1 = 'D:/xuelangAI/train_crop_feizhengchang2/'
files1 = os.listdir(file_dir1)
for i in range(0, len(files1)):
    im = Image.open(file_dir1+files1[i])
    rotate1 = im.transpose(Image.ROTATE_90)
    rotate2 = im.transpose(Image.ROTATE_180)
    rotate3 = im.transpose(Image.ROTATE_270)
    box = [rotate1, rotate2, rotate3]
    for j in range(0,3):
        name = os.path.splitext(files1[i])
        box[j].save((file_dir1+name[0]+'-%d'%j+name[1]))
    print('Rotating No.%d image'%i)

# resize 训练集图片
import os
from PIL import Image
dir1 = 'D:/xuelangAI/train_crop_feizhengchang1/'
#dir2 = 'D:/xuelangAI/train_crop_feizhengchang2/'
#dir3 = 'D:/xuelangAI/train_crop_feizhengchang3/'
#dir22 = 'D:/xuelangAI/train_resize128feizhengchang2/'
#dir33 = 'D:/xuelangAI/train_resize128feizhengchang3/'
dir11 = 'D:/xuelangAI/train_resize224feizhengchang1/'
files1 = os.listdir(dir1)
files2 = os.listdir(dir2)
files3 = os.listdir(dir3)
def img_resize(files, dir, path):
    for i in range(0,len(files)):
        im = Image.open(dir+files[i])
        img_temp = im.resize((224,224), Image.ANTIALIAS)
        img_temp.save(path+files[i], quality=100)
        print('Resizing No.%d image'%i)

img_resize(files1, dir1, dir11)
img_resize(files2, dir2, dir22)
img_resize(files3, dir3, dir33)

# resize 测试集图片
import os
from PIL import Image
dir0 = 'D:/xuelangAI/xuelang_round1_test_a_20180709/xuelang_round1_test_a_20180709/'
#dir0 = 'D:/xuelangAI/train_zhengchang/'
dir1 = 'D:/xuelangAI/test_resize896/'
files1 = os.listdir(dir0)
def img_resize(files, dir, path):
    for i in range(0,len(files)):
        im = Image.open(dir+files[i])
        img_temp = im.resize((896,896), Image.ANTIALIAS)
        img_temp.save(path+files[i], quality=100)
        print('Resizing No.%d image'%i)

img_resize(files1, dir0, dir1)

# resize 正常图片
import os
from PIL import Image
dir0 = 'D:/xuelangAI/train_zhengchang/'
dir1 = 'D:/xuelangAI/train_resize896/'
files1 = os.listdir(dir0)
def img_resize(files, dir, path):
    for i in range(0,len(files)):
        im = Image.open(dir+files[i])
        img_temp = im.resize((896,896), Image.ANTIALIAS)
        img_temp.save(path+files[i], quality=100)
        print('Resizing No.%d image'%i)

img_resize(files1, dir0, dir1)

import pickle
a = pickle.load(open('D:/xuelangAI/test_prob.pickle', 'rb'))
b = pickle.load(open('D:/xuelangAI/test_name.pickle', 'rb'))
temp = []
for i in range(a.shape[0]):
    temp.append(a[i][0])

import pandas as pd
dataframe = pd.DataFrame({'filename':b,'probability':temp})
dataframe.to_csv("D:/xuelangAI/test.csv",index=False,sep=',')


# 双边模糊
import os
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

dir0 = 'D:/xuelangAI/train_zhengchang/'
dir1 = 'D:/xuelangAI/train_resize64feizhengchang/'
dir2 = 'D:/xuelangAI/test_crop/'
files0 = os.listdir(dir0)
files1 = os.listdir(dir1)
files2 = os.listdir(dir2)
def blur(files, dir, path):
    for i in range(0,len(files)):
        im = cv2.imread(dir+files[i],0)
        blur = cv2.bilateralFilter(im,15,10,10,0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print('Blurring No.%d image'%i)


blur = cv2.bilateralFilter(im,15,10,10,0)


plt.subplot(121)
plt.imshow(im)
plt.title('Original') 
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(th3, cmap='gray')
plt.title('Blurred') 
plt.xticks([])
plt.yticks([])
plt.show()
