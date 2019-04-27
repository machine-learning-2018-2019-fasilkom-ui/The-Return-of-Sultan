# import sys
# import os
from matplotlib.legend_handler import HandlerLine2D
import numpy as np  # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color, io, exposure, img_as_uint, img_as_float
from skimage.feature import hog
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from subprocess import call
import PIL
from matplotlib import patches
import os

# Augmentation Import
from skimage.transform import rescale, rotate, resize
from skimage.util import random_noise
from scipy import ndimage
from PIL import Image, ImageOps

# dir_path = '/home/rifat/Research/ship-classification/'
# sys.path.append(dir_path)
import pickle
# from gzip_pickle import *
from gzip_pickle import *
from sklearn.model_selection import StratifiedKFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will
# list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

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

# {data:[{label, locations, scene_ids}]}

print(dataset["scene_ids"][0])

# try
# assert 1 == 0
# break
# exit()

dataset.keys()

data = np.array(dataset['data']).astype('uint8')
labels = np.array(dataset['labels']).reshape(len(dataset['labels']), 1)

# Define Positive and Negative Class
try:
    img_pos_idx = load_pickle('img_pos_idx_dev.pickle.gz')
    img_neg_idx = load_pickle('img_neg_idx_dev.pickle.gz')
except:
    img_pos_idx = []
    img_neg_idx = []
    for i in range(labels.shape[0]):
        img_neg_idx.append(i) if labels[i] == 0 else img_pos_idx.append(i)

    # Limit data size
    save_pickle(img_pos_idx[:], 'img_pos_idx_dev.pickle.gz')
    save_pickle(img_neg_idx[:], 'img_neg_idx_dev.pickle.gz')
    img_pos_idx = img_pos_idx[:]
    img_neg_idx = img_neg_idx[:]


img_pos = data[img_pos_idx]
img_neg = data[img_neg_idx]

img_length = 80
data = data.reshape(-1, 3, img_length, img_length).transpose([0, 2, 3, 1])

# print("data:",  data.shape)
# print("img_pos:",  img_pos_idx)
# print("img_neg:", img_neg_idx)

path_aug = 'augmented_data/'

if not os.path.exists(path_aug+'0/'):
    os.makedirs(path_aug+'0/')

if not os.path.exists(path_aug+'1/'):
    os.makedirs(path_aug+'1/')

rotate_list = [i*15 for i in range(1, 25)]
print(rotate_list)
for i_image in img_neg_idx:
    for i_rot in range(len(rotate_list)):
        image_aug = rotate(data[i_image], rotate_list[i_rot])
        image_aug = img_as_uint(image_aug)
        io.imsave(path_aug+'0/'+dataset["scene_ids"][i_image]+'- rotate' +
                  str(rotate_list[i_rot])+'.png', image_aug)
print("Finish augmented image(Non ship) - rotate")
for i_image in img_pos_idx:
    for i_rot in range(len(rotate_list)):
        image_aug = rotate(data[i_image], rotate_list[i_rot])
        image_aug = img_as_uint(image_aug)
        io.imsave(path_aug+'1/'+dataset["scene_ids"][i_image]+'- rotate' +
                  str(rotate_list[i_rot])+'.png', image_aug)
print("Finish augmented image(ship) - rotate")

for i_image in img_neg_idx:
    # HFlip
    image_aug = data[i_image][:, ::-1]
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'0/'+dataset["scene_ids"]
              [i_image]+'- HFlip' + '.png', image_aug)
    # VFlip
    image_aug = data[i_image][::-1, :]
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'0/'+dataset["scene_ids"]
              [i_image]+'- VFlip' + '.png', image_aug)
print("Finish augmented image(Non ship) - HFlip and VFlip")

for i_image in img_pos_idx:
    # HFlip
    image_aug = data[i_image][:, ::-1]
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'1/'+dataset["scene_ids"]
              [i_image]+'- HFlip' + '.png', image_aug)
    # VFlip
    image_aug = data[i_image][::-1, :]
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'1/'+dataset["scene_ids"]
              [i_image]+'- VFlip' + '.png', image_aug)
print("Finish augmented image(ship) - HFlip and VFlip")

for i_image in img_neg_idx:
    # Gaussian Noise
    image_aug = random_noise(
        data[i_image], mode='gaussian', seed=None, clip=True)
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'0/'+dataset["scene_ids"]
              [i_image]+'- gaussian noise' + '.png', image_aug)
    # Salt & Pepper Noise
    image_aug = random_noise(data[i_image], mode='s&p', seed=None, clip=True)
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'0/'+dataset["scene_ids"]
              [i_image]+'- s&p noise' + '.png', image_aug)
print("Finish augmented image(Non ship) - Noise")

for i_image in img_pos_idx:
    # Gaussian Noise
    image_aug = random_noise(
        data[i_image], mode='gaussian', seed=None, clip=True)
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'1/'+dataset["scene_ids"]
              [i_image]+'- gaussian noise' + '.png', image_aug)
    # Salt & Pepper Noise
    image_aug = random_noise(data[i_image], mode='s&p', seed=None, clip=True)
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'1/'+dataset["scene_ids"]
              [i_image]+'- s&p noise' + '.png', image_aug)
print("Finish augmented image(ship) - Noise")

for i_image in img_neg_idx:
    # Very Soft Blur
    image_aug = ndimage.uniform_filter(data[i_image], size=(3, 3, 1))
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'0/'+dataset["scene_ids"]
              [i_image]+'- VerySoft Blur' + '.png', image_aug)
    # Soft Blur
    image_aug = ndimage.uniform_filter(data[i_image], size=(5, 5, 1))
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'0/'+dataset["scene_ids"]
              [i_image]+'- Soft blur' + '.png', image_aug)
print("Finish augmented image(Non ship) - Blur")

for i_image in img_pos_idx:
    # Very Soft Blur
    image_aug = ndimage.uniform_filter(data[i_image], size=(3, 3, 1))
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'1/'+dataset["scene_ids"]
              [i_image]+'- VerySoft Blur' + '.png', image_aug)
    # Soft Blur
    image_aug = ndimage.uniform_filter(data[i_image], size=(5, 5, 1))
    image_aug = img_as_uint(image_aug)
    io.imsave(path_aug+'1/'+dataset["scene_ids"]
              [i_image]+'- Soft blur' + '.png', image_aug)
print("Finish augmented image(ship) - Blur")

for i_image in img_neg_idx:
    # Rescale 0.75
    im_rescaled = rescale(data[i_image], 0.75, anti_aliasing=True)
    im_pad = np.stack([np.pad(im_rescaled[:, :, c], (10,), mode='constant',
                              ) for c in range(3)], axis=2)
    image_aug = img_as_uint(im_pad)
    io.imsave(path_aug+'0/'+dataset["scene_ids"]
              [i_image]+'- Rescale_0.75' + '.png', image_aug)
print("Finish augmented image(Non ship) - Rescale")

for i_image in img_pos_idx:
    # Rescale 0.75
    im_rescaled = rescale(data[i_image], 0.75, anti_aliasing=True)
    im_pad = np.stack([np.pad(im_rescaled[:, :, c], (10,), mode='constant',
                              ) for c in range(3)], axis=2)
    image_aug = img_as_uint(im_pad)
    io.imsave(path_aug+'1/'+dataset["scene_ids"]
              [i_image]+'- Rescale_0.75' + '.png', image_aug)
print("Finish augmented image(ship) - Rescale")

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax1.imshow(data[0])

# im_rescaled = rescale(data[0], 0.75, anti_aliasing=True)
# im_pad2 = np.stack([np.pad(im_rescaled[:, :, c], (10,), mode='constant',
#                            ) for c in range(3)], axis=2)
# print(im_pad2.shape)
# ax2 = fig.add_subplot(2, 2, 2)
# ax2.imshow(im_pad2)

# im_rescaled = rescale(data[0], 0.50, anti_aliasing=True)
# im_pad3 = np.stack([np.pad(im_rescaled[:, :, c], (20,), mode='constant',
#                            ) for c in range(3)], axis=2)
# print(im_pad3.shape)
# ax3 = fig.add_subplot(2, 2, 3)
# ax3.imshow(im_pad3)

# im_rescaled = rescale(data[0], 0.25, anti_aliasing=True)
# im_pad4 = np.stack([np.pad(im_rescaled[:, :, c], (30,), mode='constant',
#                            ) for c in range(3)], axis=2)
# print(im_pad4.shape)
# ax4 = fig.add_subplot(2, 2, 4)
# ax4.imshow(im_pad4)

# im_rescaled = rescale(data[0], 0.25, anti_aliasing=True)
# im_pad = np.stack([np.pad(im_rescaled[:, :, c], (30,), mode='constant',
#                           ) for c in range(3)], axis=2)
# image_aug = img_as_uint(image_aug)
# plt.imshow(image_aug)
# io.imsave(path_aug+dataset["scene_ids"][i_image]+'- Rescale' + '.png'
# , im_pad)
