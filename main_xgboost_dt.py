# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from IPython.display import Image
from subprocess import call
import PIL
from matplotlib import patches
from gzip_pickle import *
from sklearn.model_selection import StratifiedKFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

f = open(r'input/shipsnet.json')
dataset = json.load(f)
f.close()

dataset.keys()

data = np.array(dataset['data']).astype('uint8')
img_length = 80
data = data.reshape(-1, 3, img_length, img_length).transpose([0, 2, 3, 1])

plt.imshow(data[51])

data_gray = [color.rgb2gray(i) for i in data]
# plt.imshow(data_gray[51])

ppc = 16
hog_images = []
hog_features = []
for image in data_gray:
    fd, hog_image = hog(image, orientations=16, pixels_per_cell=(
        ppc, ppc), cells_per_block=(4, 4), block_norm='L2', visualise=True)
    hog_images.append(hog_image)
    hog_features.append(fd)

plt.imshow(hog_images[51])

labels = np.array(dataset['labels']).reshape(len(dataset['labels']), 1)

clf = XGBClassifier()
hog_features = np.array(hog_features)
data_frame = np.hstack((hog_features, labels))
np.random.shuffle(data_frame)

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)
skf.get_n_splits(hog_features, labels)
print(skf)

i = 0
classification_reports = []
precision_0 = []
precision_1 = []
recall_0 = []
recall_1 = []
f_score_0 = []
f_score_1 = []
weighted_avg_f1_score = []

train_test_index = []
for train_index, test_index in skf.split(hog_features, labels):
    train_test_index.append((train_index, test_index))
    print("Fold-", (i+1))
    # print("TRAIN index:", train_index, " Size:", len(train_index))
    # print("TEST index:", test_index, " Size: ", len(test_index))
    x_train, x_test = hog_features[train_index],  hog_features[test_index]
    y_train, y_test = labels[train_index],  labels[test_index]
    # print("Labels:", labels[train_index])
    clf.fit(x_train, y_train)
    print("Finish Training")

    y_pred = clf.predict(x_test)
    y_pred_training = clf.predict(x_train)
    print("Finish Predict")

    print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
    print("Accuracy Training: "+str(accuracy_score(y_train, y_pred_training)))
    print('\n')
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports.append(report)

    f_score_0.append(report["0"]["f1-score"])
    f_score_1.append(report["1"]["f1-score"])
    weighted_avg_f1_score.append(report["weighted avg"]["f1-score"])
    i += 1


print("Final Report")
# print(classification_reports)
print("f_score_0", f_score_0)
print("f_score_1", f_score_1)
print("weighted_avg_f1_score", weighted_avg_f1_score)
print("Fold Average F-Score Class 0", sum(f_score_0)/len(f_score_0))
print("Fold Average F-Score Class 1", sum(f_score_1)/len(f_score_1))
print("Fold Average weighted_avg_f1_score", sum(
    weighted_avg_f1_score)/len(weighted_avg_f1_score))

# exit()

print("Best model")
arg_max = weighted_avg_f1_score.index(max(weighted_avg_f1_score))
print("Fold:", arg_max+1)
train_index = train_test_index[arg_max][0]
test_index = train_test_index[arg_max][1]
x_train, x_test = hog_features[train_index],  hog_features[test_index]
y_train, y_test = labels[train_index],  labels[test_index]
clf.fit(x_train, y_train.ravel())
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))



# scene = PIL.Image.open('input/scenes/lb_1.png')
scene = PIL.Image.open('case_laporan/makasar_1.jpg')
tensor = np.array(scene).astype('uint8')
width, height = scene.size
STEP_SIZE = 20
# fig = plt.figure(figsize=(16,32))
# ax = fig.add_subplot(3, 1, 1)
# ax.imshow(tensor)
# plt.show()

ships = {}

for row in range(0, height, STEP_SIZE):
    for col in range(0, width, STEP_SIZE):
        area = tensor[row:row+img_length, col:col+img_length, 0:3]
        if area.shape != (80, 80, 3):
            continue
        area = color.rgb2gray(area)
        fd, hog_image = hog(area, orientations=16, pixels_per_cell=(
            ppc, ppc), cells_per_block=(4, 4), block_norm='L2', visualize=True)
        # print("HOG-Pred")
        # fig = plt.figure(figsize=(16,32))
        # ax = fig.add_subplot(3, 1, 1)
        # ax.imshow(hog_image)
        # plt.show()
        hog_features = None
        hog_features = fd.reshape(1, len(fd))
        prediction = clf.predict(hog_features)
        # print(prediction)

        if prediction[0] == 1:
            print(f"found ship at [{row},{col}] with class {prediction}")
            ships[row, col] = prediction

fig = plt.figure(figsize=(16, 32))
ax = fig.add_subplot(3, 1, 1)

ax.imshow(tensor)

for ship in ships:
    row, col = ship
    ax.add_patch(patches.Rectangle((col, row), 80,
                                   80, edgecolor='r', facecolor='none'))

plt.show()
