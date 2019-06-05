import numpy as np
import tensorflow as tf
import os
import time
import cv2
start = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import pickle
train_data = pickle.load(open('gs://txr_data/train_data.pickle', 'rb'))
test_data = pickle.load(open('gs//txr_data/test_data.pickle', 'rb'))
label = pickle.load(open('gs//txr_data/label.pickle', 'rb'))


print('************正在打乱训练集...\n')
permutation = np.random.permutation(train_data.shape[0])
train_data = train_data[permutation, :,:]
label = label[permutation]


print('*****************开始运行CNN...\n')
# 训练的迭代次数
training_iters = 1000 
# learning rate: 降低 cost/loss/cross entropy
learning_rate = 0.0001
# batch size: 将训练图片分成一个固定的size,每个batch存储一定数量的图片
batch_size = 32 # power of 2

# MNIST data input (img shape: 28*28)
n_input = 224
# MNIST total classes (0-9 digits)
n_class = 2

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
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 224,224,3])  
ys = tf.placeholder(tf.float32, [None, n_class])
keep_prob = tf.placeholder(tf.float32)


### conv1 layer ##
W_conv1_1 = weight_variable([3,3,3,32]) 
b_conv1_1 = bias_variable([32])
h_conv1_1 = tf.nn.relu(conv2d(xs, W_conv1_1) + b_conv1_1) 

W_conv1_2 = weight_variable([3,3,32,32])
b_conv1_2 = bias_variable([32])
h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)
h_pool1 = max_pool_2x2(h_conv1_2)                                 

## conv2 layer ##
W_conv2_1 = weight_variable([3,3,32,64]) 
b_conv2_1 = bias_variable([64])
h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1)

W_conv2_2 = weight_variable([3,3,64,64])
b_conv2_2 = bias_variable([64])
h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)
h_pool2 = max_pool_2x2(h_conv2_2)                                     

## conv3 layer ##
W_conv3_1 = weight_variable([3,3,64,128])
b_conv3_1 = bias_variable([128])
h_conv3_1 = tf.nn.relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1) 

W_conv3_2 = weight_variable([3,3,128,128])
b_conv3_2 = bias_variable([128])
h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2)

W_conv3_3 = weight_variable([3,3,128,128])
b_conv3_3 = bias_variable([128])
h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2, W_conv3_3) + b_conv3_3)

h_pool3 = max_pool_2x2(h_conv3_3)     

## conv4 layer ##
W_conv4_1 = weight_variable([3,3,128,256]) # patch 3x3, in size 64, out size 128
b_conv4_1 = bias_variable([256])
h_conv4_1 = tf.nn.relu(conv2d(h_pool3, W_conv4_1) + b_conv4_1) 

W_conv4_2 = weight_variable([3,3,256,256])
b_conv4_2 = bias_variable([256])
h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)

W_conv4_3 = weight_variable([3,3,256,256])
b_conv4_3 = bias_variable([256])
h_conv4_3 = tf.nn.relu(conv2d(h_conv4_2, W_conv4_3) + b_conv4_3)

h_pool4 = max_pool_2x2(h_conv4_3)

## conv5 layer ##
W_conv5_1 = weight_variable([3,3,256,512]) 
b_conv5_1 = bias_variable([512])
h_conv5_1 = tf.nn.relu(conv2d(h_pool4, W_conv5_1) + b_conv5_1) 

W_conv5_2 = weight_variable([3,3,512,512])
b_conv5_2 = bias_variable([512])
h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, W_conv5_2) + b_conv5_2)

W_conv5_3 = weight_variable([3,3,512,512])
b_conv5_3 = bias_variable([512])
h_conv5_3 = tf.nn.relu(conv2d(h_conv5_2, W_conv5_3) + b_conv5_3)

h_pool5 = max_pool_2x2(h_conv5_3)

## fc1 layer ##
W_fc1 = get_weight([7*7*512,512],0.003)
b_fc1 = bias_variable([512])
h_pool5_flat = tf.reshape(h_pool5, [-1, 7*7*512])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = get_weight([512, 128],0.003)
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


W_fc3 = get_weight([128, n_class],0.003)
b_fc3 = bias_variable([n_class])
prediction = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

cross_entropy = -tf.reduce_mean(ys*tf.log(tf.clip_by_value(prediction,1e-15,1.0)))
tf.add_to_collection('losses', cross_entropy)
# get_collection()返回一个列表，这个列表是所有这个集合中的元素，在本样例中这些元素就是损失函数的不同部分，将他们加起来就是最终的损失函数
cost = tf.add_n(tf.get_collection('losses'))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


pred = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float')) 


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(200):
    for batch in range(len(train_data)//batch_size+1):
        batch_x = train_data[batch*batch_size:min((batch+1)*batch_size,len(train_data))]
        batch_y = label[batch*batch_size:min((batch+1)*batch_size,len(label))]    
        opt = sess.run(optimizer, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.4})
        loss, acc = sess.run([cost, accuracy], feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.4}) 
    
    pb = softmax_accuracy(test_data[0:32])
    print('前200张图片的概率为：\n', pb)
    time_temp = time.time()
    print('已耗时: %s 分钟...\n'%((time_temp-start)/60))
    print('\n'+"Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

print('*******************训练结束,放入测试集...\n')
prob_list = []
for batch_test in range(len(test_data)//batch_size):
    batch = test_data[batch_test*batch_size:min((batch_test+1)*batch_size, len(test_data))]
    prob = softmax_accuracy(batch)
    prob_list.append(prob)
print('******************测试集长度为：%d\n'%(len(prob_list)))

# 提取图片的概率
temp = []
for i in range(0, len(prob_list)):
    for j in range(0,batch_size):
      
        tt = prob_list[i][j]
        #print('**************提取第%d个batch，第%d张图片的softmax概率'%(i,j))
        temp.append(tt[0])
#for m in range(0, 32):
#    tt = prob_list[len(prob_list)-1][m]
#    temp.append(tt[0])
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
result1 = open('gs://txr_data/test_prob.pickle', 'wb')
pickle.dump(prob_temp, result1)
result1.close()

result2 = open('gs://txr_data/test_name.pickle', 'wb')
pickle.dump(names, result2)
result2.close()

print('*******************生成提交文件...\n')
print('文件名长度：%d, 概率长度：%d'%(len(names), len(prob_temp)))
import pandas as pd
dataframe = pd.DataFrame({'filename':names,'probability':prob_temp})
dataframe.to_csv("gs://txr_data/test2.csv",index=False,sep=',')

end = time.time()

print ('***************程序运行结束，总耗时%s分钟\n'%((end-start)/60))




