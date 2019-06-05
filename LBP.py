from skimage import io, color, feature
import matplotlib.pyplot as plt
import numpy as np
def convert_to_gray(f, **args):
    image = io.imread(f)
    image = color.rgb2gray(image)
    return image

train_dir0 = 'D:/xuelangAI/train_正常/'
train_dir1 = 'D:/xuelangAI/train_非正常/'
train0_path = train_dir0 + '/*.jpg'
train1_path = train_dir1 + '/*.jpg'
images0 = io.ImageCollection(train0_path, load_func=convert_to_gray)
images1 = io.ImageCollection(train1_path, load_func=convert_to_gray)
#original = io.ImageCollection(train0_path)
size0 = int(len(images0))
size1 = int(len(images1))
train_data = []
label = []

def trainingSet(image, num):
    if (num == 0):
        for i in range(0,size1):
            lbp = feature.local_binary_pattern(image[i], 63, 8, method='uniform')
            n_bins = int(lbp.max() + 1)
            h = np.histogram(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins))
            train_data.append(h[0])           
            label.append('0')
            print('Processed training image', i)
            #imsave('D:/xuelangAI/lbp_正常/%d.jpg'%i, lbp)
    else:
        for i in range(0,size1):
            lbp = feature.local_binary_pattern(image[i], 63, 8, method='uniform')
            n_bins = int(lbp.max() + 1)
            h = np.histogram(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins))
            train_data.append(h[0])
            label.append('1')
            print('Processed training image', i)
            #imsave('D:/xuelangAI/lbp_非正常/%d.jpg'%i, lbp)

trainingSet(images0, 0)
trainingSet(images1, 1)

import pickle
file1 = open('D:/xuelangAI/train_data.pickle', 'wb')
pickle.dump(train_data, file1)
file1.close()

f = open('D:/xuelangAI/train_data.pickle', 'rb')
train_data = pickle.load(f)

file2 = open('train_label.pickle', 'wb')
pickle.dump(label, file2)
file2.close()

test_data = []
def testingSet(image):
    for i in range(size1, size1+100):
        lbp = feature.local_binary_pattern(image[i], 63, 8, method='uniform')
        n_bins = int(lbp.max() + 1)
        h = np.histogram(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins))
        test_data.append(h[0])
        print('Processed testing image', i)

testingSet(images0)

for i in range(0,size1):
    label.append('0')

for i in range(size1, len(train_data)):
    label.append('1')

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50,50,50,), random_state=1, max_iter=10, learning_rate_init=.1)
clf.fit(train_data, label)
print(clf.predict_proba(test_data))
    #clf.predict(h[0].reshape(1,-1))
    #clf.predict_proba(h[0].reshape(1,-1))



#plt.bar(h[1][:-1], h[0], width = 1)
#plt.show()

#plt.subplot(2,2,1)
#plt.title('original image')
#plt.imshow(images1[1], cmap='gray')

#plt.subplot(2,2,2)
#plt.title('LBP image')
#plt.imshow(lbp, cmap='gray')
#plt.show()
