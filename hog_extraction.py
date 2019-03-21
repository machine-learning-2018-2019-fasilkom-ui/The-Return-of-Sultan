import numpy as np # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
import pickle

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

# f = open(r'input/shipsnet.json')
# dataset = json.load(f)
# f.close()

dataset.keys()
print(dataset.keys())

data = np.array(dataset['data']).astype('uint8')

data.shape
print(data.shape)
img_length = 80
data = data.reshape(-1,3,img_length,img_length).transpose([0,2,3,1])
data.shape
labels =  np.array(dataset['labels']).reshape(len(dataset['labels']),1)
print(labels)

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
plt.imshow(data[1001])

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
plt.imshow(data_gray[1001])

plt.subplot(2, 2, 2)
plt.imshow(data_gray[2828])

plt.subplot(2, 2, 3)
plt.imshow(data_gray[2813])

plt.subplot(2, 2, 4)
plt.imshow(data_gray[3000])
plt.show()

ppc = 16
hog_images = []
hog_features = []
for image in data_gray:
    fd,hog_image = hog(image, orientations=16, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
    hog_images.append(hog_image)
    hog_features.append(fd)


fig = plt.figure(figsize=(8, 8))
fig.suptitle('HOG Features (O=16) in Ship Images in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(hog_images[0])

plt.subplot(2, 2, 2)
plt.imshow(hog_images[1])

plt.subplot(2, 2, 3)
plt.imshow(hog_images[2])

plt.subplot(2, 2, 4)
plt.imshow(hog_images[3])
plt.show()


fig = plt.figure(figsize=(8, 8))
fig.suptitle('HOG Features (O=16) in Grayscale Non-ship Images in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(hog_images[1001])

plt.subplot(2, 2, 2)
plt.imshow(hog_images[2828])

plt.subplot(2, 2, 3)
plt.imshow(hog_images[2813])

plt.subplot(2, 2, 4)
plt.imshow(hog_images[3000])
plt.show()

ppc = 16
hog_images = []
hog_features = []
for image in data_gray:
    fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
    hog_images.append(hog_image)
    hog_features.append(fd)


fig = plt.figure(figsize=(8, 8))
fig.suptitle('HOG Features (O=8) in Ship Images in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(hog_images[0])

plt.subplot(2, 2, 2)
plt.imshow(hog_images[1])

plt.subplot(2, 2, 3)
plt.imshow(hog_images[2])

plt.subplot(2, 2, 4)
plt.imshow(hog_images[3])
plt.show()


fig = plt.figure(figsize=(8, 8))
fig.suptitle('HOG Features (O=8) in Grayscale Non-ship Images in Dataset', fontsize=20)
plt.subplot(2, 2, 1)
plt.imshow(hog_images[1001])

plt.subplot(2, 2, 2)
plt.imshow(hog_images[2828])

plt.subplot(2, 2, 3)
plt.imshow(hog_images[2813])

plt.subplot(2, 2, 4)
plt.imshow(hog_images[3000])
plt.show()

