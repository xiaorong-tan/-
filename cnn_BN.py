import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import cv2
start = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 使用第一个GPU

dir0 = 'D:/xuelangAI/train_crop_zhengchang64/'
dir1 = 'D:/xuelangAI/train_resize64feizhengchang1/'
#dir2 = 'D:/xuelangAI/train_resize64feizhengchang2/'
#dir3 = 'D:/xuelangAI/train_resize64feizhengchang3/'
dir4 = 'D:/xuelangAI/test_crop64/'
#dir0 = 'D:/xuelangAI/train_resize64正常/'
#dir1 = 'D:/xuelangAI/train_resize64非正常/'
files0 = os.listdir(dir0)
files1 = os.listdir(dir1)
#files2 = os.listdir(dir2)
#files3 = os.listdir(dir3)
files4 = os.listdir(dir4)
train_data = []
test_data = []
label=[]

# 将图片转为数组
def load_data(imgName, num, path, a, b):
    for i in range(0,num):
        im = Image.open(path+imgName[i])
        arr = np.asarray(im,dtype='float32')
        train_data.append(arr)
        label.append([a,b])
        if i % 50 == 0:
            print('Converting training images No.%d image to array'%i)

# 将训练集的图片转为数组
load_data(files0, len(files0), dir0, 0.0, 1.0)
load_data(files1, len(files1), dir1, 1.0, 0.0)
#load_data(files2, len(files2), dir2, 1.0, 0.0)
#load_data(files3, len(files3), dir3, 1.0, 0.0)

# 将测试集的图片转为数组
for i in range(0, len(files4)):
    im = Image.open(dir4+files4[i])
    arr = np.asarray(im, dtype='float32')
    test_data.append(arr)
    if i % 50 == 0:
        print('Converting testing images No.%d image to array'%i)

# 把list 转为 数组
label = np.array(label)
train_data = np.array(train_data)
test_data = np.array(test_data)

# 归一化
def norm(size, dataset):
    for i in range(0, size):
        dataset[i] = (dataset[i] - np.min(dataset[i])) / (np.max(dataset[i] - np.min(dataset[i])))

norm(len(train_data), train_data)
norm(len(test_data), test_data)

print('************正在打乱训练集...\n')
permutation = np.random.permutation(train_data.shape[0])
train_data = train_data[permutation, :,:]
label = label[permutation]


#X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.25, random_state=0)
#X_train = np.array(X_train)
#X_test = np.array(X_test)
#y_train = np.array(y_train)
#y_test = np.array(y_test)

print('*****************开始运行CNN...\n')
# 训练的迭代次数
training_iters = 1000 
# learning rate: 降低 cost/loss/cross entropy
learning_rate = 0.0001
# batch size: 将训练图片分成一个固定的size,每个batch存储一定数量的图片
batch_size = 64 # power of 2

# MNIST data input (img shape: 28*28)
n_input = 64
# MNIST total classes (0-9 digits)
n_class = 2


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)

def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.001), dtype=tf.float32) # 生成一个变量
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var)) # add_to_collection()函数将新生成变量的L2正则化损失加入集合losses
    return var # 返回生成的变量

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 64,64,3])   # 32x32
ys = tf.placeholder(tf.float32, [None, n_class])
is_training = tf.placeholder(tf.bool)


def cnn_with_BN(x, is_training):
    
    W_conv1 = weight_variable([3,3,3,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.layers.batch_normalization(conv2d(x, W_conv1)+b_conv1, training=is_training)
    h_conv1 = tf.nn.relu(h_conv1) 
    h_pool1 = max_pool_2x2(h_conv1)                                 

    W_conv2 = weight_variable([3,3,32,64]) 
    b_conv2 = bias_variable([64])
    h_conv2 = tf.layers.batch_normalization(conv2d(h_pool1, W_conv2)+b_conv2, training=is_training)
    h_conv2 = tf.nn.relu(h_conv2) 
    h_pool2 = max_pool_2x2(h_conv2)                                         

    W_conv3 = weight_variable([3,3,64,128]) # patch 3x3, in size 64, out size 128
    b_conv3 = bias_variable([128])
    h_conv3 = tf.layers.batch_normalization(conv2d(h_pool2, W_conv3)+b_conv3, training=is_training)
    h_conv3 = tf.nn.relu(h_conv3) 
    h_pool3 = max_pool_2x2(h_conv3)                                       

    W_fc1 = get_weight([8*8*128,128],0.003)
    b_fc1 = bias_variable([128])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*128])
    h_pool3_flat = tf.layers.batch_normalization(tf.matmul(h_pool3_flat, W_fc1)+b_fc1, training=is_training)
    h_fc1 = tf.nn.tanh(h_pool3_flat)

    W_fc2 = get_weight([128, 64],0.003)
    b_fc2 = bias_variable([64])
    h_fc2 = tf.layers.batch_normalization(tf.matmul(h_fc1, W_fc2)+b_fc2, training=is_training)
    h_fc2 = tf.nn.tanh(h_fc2)

    W_fc3 = get_weight([64, n_class],0.003)
    b_fc3 = bias_variable([n_class])
    pred = tf.matmul(h_fc2, W_fc3) + b_fc3
    return pred

out = cnn_with_BN(xs, is_training)
prediction = tf.nn.softmax(out)

cross_entropy = -tf.reduce_mean(ys*tf.log(tf.clip_by_value(prediction,1e-15,1.0)))
tf.add_to_collection('losses', cross_entropy)
# get_collection()返回一个列表，这个列表是所有这个集合中的元素，在本样例中这些元素就是损失函数的不同部分，将他们加起来就是最终的损失函数
cost = tf.add_n(tf.get_collection('losses'))

# 通知Tensorflow在训练时要更新均值和方差的分布
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    for batch in range(len(train_data)//batch_size+1):
        batch_x = train_data[batch*batch_size:min((batch+1)*batch_size,len(train_data))]
        batch_y = label[batch*batch_size:min((batch+1)*batch_size,len(label))]    
        opt = sess.run(optimizer, feed_dict={xs: batch_x, ys: batch_y, is_training:True})
        loss, acc = sess.run([cost, accuracy], feed_dict={xs: batch_x, ys: batch_y, is_training:False})

    #pb1 = tf.nn.softmax(cnn_with_BN(xs, is_training))
    pb = sess.run(prediction, feed_dict={xs:test_data[0:200], is_training:False})
    print('前200张图片的概率为：\n', pb)
    time_temp = time.time()
    print('已耗时: %s 分钟...\n'%((time_temp-start)/60))
    print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc) + '\n')

print('*******************训练结束,放入测试集...\n')
prob_list = []
for batch_test in range(len(test_data)//batch_size+1):
    batch = test_data[batch_test*batch_size:min((batch_test+1)*batch_size, len(test_data))]
    prob = sess.run(prediction, feed_dict={xs:batch, is_training:False})
    prob_list.append(prob)
print('******************测试集长度为：%d\n'%(len(prob_list)))

# 提取图片的概率
temp = []
for i in range(0, len(prob_list)-1):
    for j in range(0,batch_size):
      
        tt = prob_list[i][j]
        #print('**************提取第%d个batch，第%d张图片的softmax概率'%(i,j))
        temp.append(tt[0])

for m in range(0, 32):
    tt = prob_list[len(prob_list)-1][m]
    temp.append(tt[0])
print('***************概率总长度：%d\n'%len(temp))
# 提取概率最大的
m = 0
prob_temp = []
for j in range(0, int(len(temp)/16)):
    t = temp[0+m:m+16]
    prob_temp.append(max(t))
    print('**************第%d张图片的最大概率为：%s'%(j,max(t)))
    m = m + 16

# 提取测试集的文件名
names = []
names_temp = []
for i in range(0, len(files4), 16):
    names_temp.append(files4[i])
# 删除文件名中的多余字符
for i in range(0, len(names_temp)):
    names.append(names_temp[i].replace('-0',''))

# 保存概率和文件名
print('\n*********************************保存概率和文件名...\n')
import pickle
result1 = open('D:/xuelangAI/test_prob.pickle', 'wb')
pickle.dump(prob_temp, result1)
result1.close()

result2 = open('D:/xuelangAI/test_name.pickle', 'wb')
pickle.dump(names, result2)
result2.close()

print('*******************生成提交文件...\n')
print('文件名长度：%d, 概率长度：%d'%(len(names), len(prob_temp)))
import pandas as pd
dataframe = pd.DataFrame({'filename':names,'probability':prob_temp})
dataframe.to_csv("D:/xuelangAI/test2.csv",index=False,sep=',')

end = time.time()

print ('***************程序运行结束，总耗时%s分钟\n'%((end-start)/60))




