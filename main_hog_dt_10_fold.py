# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
import pickle
from gzip_pickle import *
from sklearn.model_selection import StratifiedKFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

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
    save_pickle(img_pos_idx[:100], 'img_pos_idx_dev.pickle.gz')
    save_pickle(img_neg_idx[:300], 'img_neg_idx_dev.pickle.gz')
    img_pos_idx = img_pos_idx[:100]
    img_neg_idx = img_neg_idx[:300]


img_pos = data[img_pos_idx]
img_neg = data[img_neg_idx]

# print(len(img_pos_idx))
# print(len(img_neg_idx))
# print(img_pos_idx[:4])
# print(img_neg_idx[:4])
# print(len(img_pos_idx))
# print(len(img_neg_idx))

# exit()

img_length = 80
data = data.reshape(-1, 3, img_length, img_length).transpose([0, 2, 3, 1])

# Set load data using certain index
# chosen_index = img_pos_idx + img_neg_idx
# data = data[chosen_index]
# labels = labels[chosen_index]

# print(data.shape)
# plt.imshow(data[0])
# plt.imshow(data[1001])
# plt.show()
# exit()

data.shape
img_length = 80
data = data.reshape(-1, 3, img_length, img_length).transpose([0, 2, 3, 1])

data.shape
# plt.imshow(data[5])

data_gray = [color.rgb2gray(i) for i in data]
# plt.imshow(data_gray[5])

ppc = 16
hog_images = []
hog_features = []
for image in data_gray:
    fd, hog_image = hog(image, orientations=16, pixels_per_cell=(
        ppc, ppc), cells_per_block=(4, 4), block_norm='L2', visualize=True)
    hog_images.append(hog_image)
    hog_features.append(fd)

# plt.imshow(hog_images[51])

# labels = np.array(dataset['labels']).reshape(len(dataset['labels']), 1)

clf = tree.DecisionTreeClassifier(random_state=17)


hog_features = np.array(hog_features)
data_frame = np.hstack((hog_features, labels))
np.random.shuffle(data_frame)


# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)
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
    print("TRAIN index:", train_index, " Size:", len(train_index))
    print("TEST index:", test_index, " Size: ", len(test_index))
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
# print("f_score_0", f_score_0)
# print("f_score_1", f_score_1)
# print("weighted_avg_f1_score", weighted_avg_f1_score)
# print("Fold Average F-Score Class 0", sum(f_score_0)/len(f_score_0))
# print("Fold Average F-Score Class 1", sum(f_score_1)/len(f_score_1))
# print("Fold Average weighted_avg_f1_score", sum(f_score_1)/len(f_score_1))

percentage = 80
partition = int(len(hog_features)*percentage/100)
print("Partition:"+str(partition))

x_train, x_test = data_frame[:partition, :-1],  data_frame[partition:, :-1]
y_train, y_test = data_frame[:partition, -
                             1:].ravel(), data_frame[partition:, -1:].ravel()

clf.fit(x_train, y_train)
print("Finish Training")

y_pred = clf.predict(x_test)
y_pred_training = clf.predict(x_train)
print("Finish Predict")

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print("Accuracy Training: "+str(accuracy_score(y_train, y_pred_training)))
print('\n')
print(classification_report(y_test, y_pred))
