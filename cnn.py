import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import time
import cv2
start = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 使用第一个GPU

dir0 = 'D:/xuelangAI/train_crop_zhengchang64/'
dir1 = 'D:/xuelangAI/train_resize64feizhengchang1/'
#dir2 = 'D:/xuelangAI/train_resize100feizhengchang2/'
#dir3 = 'D:/xuelangAI/train_resize100feizhengchang3/'
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
        im = cv2.imread(path+imgName[i])
        train_data.append(im)
        label.append([a,b])
        if i % 50 == 0:
            print('读入训练集第 No.%d 图片'%i)

# 将训练集的图片转为数组
load_data(files0, len(files0), dir0, 0.0, 1.0)
load_data(files1, len(files1), dir1, 1.0, 0.0)
#load_data(files2, len(files2), dir2, 1.0, 0.0)
#load_data(files3, len(files3), dir3, 1.0, 0.0)

# 将测试集的图片转为数组
for i in range(0, len(files4)):
    im = cv2.imread(dir4+files4[i])
    test_data.append(im)
    if i % 50 == 0:
        print('读入测试集第 No.%d 图片'%i)

# 把list 转为 数组
label = np.array(label)
train_data = np.array(train_data)
test_data = np.array(test_data)

# 双边模糊
#def blur(files, dir, a, b):
#    for i in range(0,len(files)):
#        im = cv2.imread(dir+files[i],0)
#        blur = cv2.bilateralFilter(im,15,10,10,0)
#        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        train_data.append(th3)
#        label.append([a,b])
#        if i % 50 == 0:
#            print('Blurring training images No.%d image to array'%i)

#for i in range(0, len(files4)):
#    im = cv2.imread(dir4+files4[i],0)
#    blur2 = cv2.bilateralFilter(im,15,10,10,0)
#    ret3,th2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    test_data.append(th2)
#    if i % 50 == 0:
#        print('Blurring testing images No.%d image to array'%i)

#blur(files0, dir0, 0.0, 1.0)
#blur(files1, dir1, 1.0, 0.0)
#blur(files2, dir2, 1.0, 0.0)
#blur(files3, dir3, 1.0, 0.0)
#label = np.array(label)
#train_data = np.array(train_data)
#test_data = np.array(test_data)
#train_data = train_data.reshape(-1, 64, 64, 1)
#test_data = test_data.reshape(-1, 64, 64, 1)

# 归一化
#def norm(size, dataset):
#    for i in range(0, size):
#        dataset[i] = (dataset[i] - np.min(dataset[i])) / (np.max(dataset[i] - np.min(dataset[i])))

# 标准化
print('*********************进行图片标准化...\n')
def norm(size, dataset):
    for i in range(0, size):
        dataset[i] = (dataset[i] - np.mean(dataset[i])) / np.std(dataset[i])

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


#x = tf.placeholder('float', [None, 64,64,3])
#y = tf.placeholder('float', [None, 2])

#def conv2d(x, W):
#    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#def maxpool2d(x):
#    #                        size of window         movement of window
#    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#def softmax_accuracy(v_xs):
#    global prediction
#    y_prediction = sess.run(prediction, feed_dict={x: v_xs, keep_prob: 1})
#    return y_prediction

#def convolutional_neural_network(x):
#    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,32], mean=0, stddev=0.01)),
#               'W_conv2':tf.Variable(tf.random_normal([3,3,32,64],mean=0, stddev=0.01)),
#               'W_conv3':tf.Variable(tf.random_normal([3,3,64,128],mean=0, stddev=0.01)),
#               'W_fc':tf.Variable(tf.random_normal([8*8*128,128],mean=0, stddev=0.01)),
#               'W_fc1':tf.Variable(tf.random_normal([128,64],mean=0, stddev=0.01)),
#               'out':tf.Variable(tf.random_normal([64, n_classes],mean=0, stddev=0.01))}

#    biases = {'b_conv1':tf.Variable(tf.random_normal([32],mean=0, stddev=0.01)),
#               'b_conv2':tf.Variable(tf.random_normal([64],mean=0, stddev=0.01)),
#               'b_conv3':tf.Variable(tf.random_normal([128],mean=0, stddev=0.01)),
#               'b_fc':tf.Variable(tf.random_normal([128],mean=0, stddev=0.01)),
#               'b_fc1':tf.Variable(tf.random_normal([64],mean=0, stddev=0.01)),
#               'out':tf.Variable(tf.random_normal([n_classes],mean=0, stddev=0.01))}

#    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
#    conv1 = maxpool2d(conv1)
    
#    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
#    conv2 = maxpool2d(conv2)

#    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
#    conv3 = maxpool2d(conv3)

#    fc = tf.reshape(conv3,[-1, 8*8*128])
#    fc = tf.nn.tanh(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
#    fc = tf.nn.dropout(fc, keep_rate)

#    fc1 = tf.nn.tanh(tf.matmul(fc, weights['W_fc1'])+biases['b_fc1'])
#    fc1 = tf.nn.dropout(fc1, keep_rate)

#    output = tf.matmul(fc1, weights['out'])+biases['out']

#    return output

def softmax_accuracy(v_xs):
    global prediction
    y_prediction = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    return y_prediction

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01), dtype=tf.float32) # 生成一个变量
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var)) # add_to_collection()函数将新生成变量的L2正则化损失加入集合losses
    return var # 返回生成的变量

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 64,64,3])   # 32x32
ys = tf.placeholder(tf.float32, [None, n_class])
keep_prob = tf.placeholder(tf.float32)


### conv1 layer ##
W_conv1 = weight_variable([3,3,3,32]) # patch 3x3, in size 1, out size 32
#W_conv1 = tf.get_variable("weights1", shape=[3,3,3,32],
#                    initializer=tf.glorot_uniform_initializer())
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1) # output size 32x32x32
h_pool1 = max_pool_2x2(h_conv1)                                 

## conv2 layer ##
W_conv2 = weight_variable([3,3,32,64]) # patch 3x3, in size 32, out size 64
#W_conv2 = tf.get_variable("weights2", shape=[3,3,32,64],
#                    initializer=tf.glorot_uniform_initializer())
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 16x16x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 8x8x64

## conv3 layer ##
W_conv3 = weight_variable([3,3,64,128]) # patch 3x3, in size 64, out size 128
#W_conv3 = tf.get_variable("weights3", shape=[3,3,64,128],
#                    initializer=tf.glorot_uniform_initializer())
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) # output size 8x18x64
h_pool3 = max_pool_2x2(h_conv3)                                         # output size 4x4x128

## fc1 layer ##
W_fc1 = get_weight([8*8*128,128],0.003)
#W_fc1 = tf.get_variable("weights4", shape=[13*13*128,128],
#                    initializer=tf.glorot_uniform_initializer())
b_fc1 = bias_variable([128])
h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*128])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = get_weight([128, 64],0.003)
#W_fc2 = tf.get_variable("weights5", shape=[128, 64],
#                    initializer=tf.glorot_uniform_initializer())
b_fc2 = bias_variable([64])
h_fc2 = tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


W_fc3 = get_weight([64, n_class],0.003)
#W_fc3 = tf.get_variable("weights6", shape=[64, n_class],
#                    initializer=tf.glorot_uniform_initializer())
b_fc3 = bias_variable([n_class])
prediction = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# the error between prediction and real data
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                              reduction_indices=[1]))       # loss
cross_entropy = -tf.reduce_mean(ys*tf.log(tf.clip_by_value(prediction,1e-15,1.0)))
tf.add_to_collection('losses', cross_entropy)
# get_collection()返回一个列表，这个列表是所有这个集合中的元素，在本样例中这些元素就是损失函数的不同部分，将他们加起来就是最终的损失函数
cost = tf.add_n(tf.get_collection('losses'))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#beta = 0.01
#pred = convolutional_neural_network(x)

pred = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
#prediction = tf.nn.softmax(pred)
#cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float')) 


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    for batch in range(len(train_data)//batch_size+1):
        batch_x = train_data[batch*batch_size:min((batch+1)*batch_size,len(train_data))]
        batch_y = label[batch*batch_size:min((batch+1)*batch_size,len(label))]    
        opt = sess.run(optimizer, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.4})
        loss, acc = sess.run([cost, accuracy], feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.4}) 
    
    pb = softmax_accuracy(test_data[0:32])
    print('前32张图片的概率为：\n', pb)
    time_temp = time.time()
    print('已耗时: %s 分钟...\n'%((time_temp-start)/60))
    print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    #print('神经网络输出结果:\n', sess.run(pred, feed_dict={x: test_data[1:20]}))
    #test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: X_test,y : y_test})
    #print("Testing Accuracy:","{:.5f}".format(test_acc))

print('*******************训练结束,放入测试集...\n')
prob_list = []
for batch_test in range(len(test_data)//batch_size+1):
    batch = test_data[batch_test*batch_size:min((batch_test+1)*batch_size, len(test_data))]
    prob = softmax_accuracy(batch)
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



