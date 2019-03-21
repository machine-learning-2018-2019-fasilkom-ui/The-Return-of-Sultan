import numpy as np # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import local_binary_pattern
import pickle
from sklearn import tree
from sklearn.metrics import classification_report,accuracy_score

# Load Ship Dataset
try:
    pickle_in = open("dataset.pickle", "rb")
    dataset = pickle.load(pickle_in)
    pickle_in.detach()
except:
    f = open(r'input/shipsnet.json')
    dataset = json.load(f)
    f.close()
    pickle_out = open("dataset.pickle", "wb")
    pickle.dump(dataset, pickle_out)
    pickle_out.close()

# settings for LBP
radius = 3
n_points = 8 * radius

dataset.keys()
print(dataset.keys())

data = np.array(dataset['data']).astype('uint8')

data.shape
print(data.shape)
img_length = 80
data = data.reshape(-1,3,img_length,img_length).transpose([0,2,3,1])
data.shape

fig = plt.figure(figsize=(8, 8))
fig.suptitle('Image of Ships in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(data[0])

plt.subplot(2, 2, 2)
plt.imshow(data[1])

plt.subplot(2, 2, 3)
plt.imshow(data[2])

plt.subplot(2, 2, 4)
plt.imshow(data[3])
plt.show()

fig = plt.figure(figsize=(8, 8))
fig.suptitle('Image of Non-Ships in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(data[99])

plt.subplot(2, 2, 2)
plt.imshow(data[2828])

plt.subplot(2, 2, 3)
plt.imshow(data[2813])

plt.subplot(2, 2, 4)
plt.imshow(data[3000])
plt.show()

data_gray = [ color.rgb2gray(i) for i in data]
fig = plt.figure(figsize=(8, 8))
fig.suptitle('Grayscale of Ship Images in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(data_gray[0])

plt.subplot(2, 2, 2)
plt.imshow(data_gray[1])

plt.subplot(2, 2, 3)
plt.imshow(data_gray[2])

plt.subplot(2, 2, 4)
plt.imshow(data_gray[3])
plt.show()

fig = plt.figure(figsize=(8, 8))
fig.suptitle('Grayscale of Non-Ship Images in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(data_gray[99])

plt.subplot(2, 2, 2)
plt.imshow(data_gray[2828])

plt.subplot(2, 2, 3)
plt.imshow(data_gray[2813])

plt.subplot(2, 2, 4)
plt.imshow(data_gray[3000])
plt.show()

ppc = 16
lbp_images = []
lbp_features = []
for image in data_gray:
    lbp_image = local_binary_pattern(image, n_points, radius)
    lbp_images.append(lbp_image)
    # lbp_features.append(fd)


fig = plt.figure(figsize=(8, 8))
fig.suptitle('LBP Features (R=3, N_Points=8*R) in Ship Images in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(lbp_images[0])

plt.subplot(2, 2, 2)
plt.imshow(lbp_images[1])

plt.subplot(2, 2, 3)
plt.imshow(lbp_images[2])

plt.subplot(2, 2, 4)
plt.imshow(lbp_images[3])
plt.show()


fig = plt.figure(figsize=(8, 8))
fig.suptitle('LBP Features (R=3, N_Points=8*R) in Grayscale Non-ship Images in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(lbp_images[99])

plt.subplot(2, 2, 2)
plt.imshow(lbp_images[2828])

plt.subplot(2, 2, 3)
plt.imshow(lbp_images[2813])

plt.subplot(2, 2, 4)
plt.imshow(lbp_images[3000])
plt.show()

labels =  np.array(dataset['labels']).reshape(len(dataset['labels']),1)
print("sdfsdf")

clf = tree.DecisionTreeClassifier(random_state=17)


lbp_features = np.array(lbp_image)
data_frame = np.hstack((lbp_image,labels))
np.random.shuffle(data_frame)
print("sdfsdf")

percentage = 80
partition = int(len(lbp_features)*percentage/100)
print("Partition:"+str(partition))

x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()

clf.fit(x_train,y_train)
print("Finish Training")

y_pred = clf.predict(x_test)
print("Finish Predict")

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))