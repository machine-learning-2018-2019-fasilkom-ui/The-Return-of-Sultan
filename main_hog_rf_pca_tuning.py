# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# import sys
# import os
from matplotlib.legend_handler import HandlerLine2D
import numpy as np  # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from subprocess import call
import PIL
from matplotlib import patches
from time import time

# dir_path = '/home/rifat/Research/ship-classification/'
# sys.path.append(dir_path)
import pickle
# from gzip_pickle import *
from gzip_pickle import *
from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA

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


# exit()

img_length = 80
data = data.reshape(-1, 3, img_length, img_length).transpose([0, 2, 3, 1])

# Set load data using certain index
chosen_index = img_pos_idx + img_neg_idx
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

# clf = tree.DecisionTreeClassifier(random_state=17)
clf = RandomForestClassifier(random_state=17)
print(clf)


hog_features = np.array(hog_features)
data_frame = np.hstack((hog_features, labels))
# np.random.shuffle(data_frame)


# # Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    hog_features, labels.ravel(), test_size=0.2, random_state=17)

# Compute a PCA
print("Features dimension:", X_train[0].shape)
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# apply PCA transformation
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

X_train = X_train_pca
X_test = X_test_pca



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=20)]
n_estimators
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
# max_depth.append(None)
# Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
# Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)

# min_impurity_decrease
min_impurity_decrease = np.linspace(0.0, 0.1, 10, endpoint=True)

# Method of selecting samples for training each tree
# bootstrap = [True, False]
bootstrap = [True]
class_weight = ["balanced"]

# Create the random grid
tuned_parameters = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    # 'min_impurity_decrease': min_impurity_decrease,
                    'bootstrap': bootstrap,
                    'class_weight': class_weight}


# Best param ROC AUC
# {'min_samples_split': 0.2, 'min_samples_leaf': 0.1, 'max_features': 'log2', 'max_depth': 12.0, 'criterion': 'gini', 'class_weight': 'balanced'}


scores = ['recall_macro']

best_estimator = []
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    start_time = time()
    n_iter = 100
    clf = RandomizedSearchCV(estimator=RandomForestClassifier(),
                             param_distributions=tuned_parameters, cv=3,
                             scoring=score, n_iter=n_iter, n_jobs=-1,
                             verbose=2, random_state=17)
    # clf = GridSearchCV(estimator=RandomForestClassifier(),
    #                    param_grid=tuned_parameters, cv=3,
    #                    scoring=score, n_jobs=-1,
    #                    verbose=2)
    clf.fit(X_train, y_train)

    end_time = time()
    seconds_elapsed = end_time - start_time
    print("Time elapsed:", seconds_elapsed)

    best_estimator.append(clf.best_estimator_)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        break
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full training set.")
    print()
    y_true, y_pred = y_train, clf.predict(X_train)
    print(classification_report(y_true, y_pred))
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
# exit()

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)
skf.get_n_splits(hog_features, labels)
print(skf)

X_train, X_test, y_train, y_test = train_test_split(
    hog_features, labels.ravel(), test_size=0.8, random_state=0)



# clf = best_estimator[0]
scene = PIL.Image.open('Indonesia_res_1080/makasar_1.jpg')

tensor = np.array(scene).astype('uint8')
width, height = scene.size
STEP_SIZE = 20
fig = plt.figure(figsize=(16, 32))
ax = fig.add_subplot(3, 1, 1)
ax.imshow(tensor)
plt.show()

ships = {}

for row in range(0, height, STEP_SIZE):
    for col in range(0, width, STEP_SIZE):
        area = tensor[row:row+img_length, col:col+img_length, 0:3]
        if area.shape != (80, 80, 3):
            continue
        area = color.rgb2gray(area)
        fd, hog_image = hog(area, orientations=16,
                            pixels_per_cell=(ppc, ppc),
                            cells_per_block=(4, 4),
                            block_norm='L2', visualize=True)
        # print("HOG-Pred")
        # fig = plt.figure(figsize=(16,32))
        # ax = fig.add_subplot(3, 1, 1)
        # ax.imshow(hog_image)
        # plt.show()
        hog_features = None
        hog_features = fd.reshape(1, len(fd))
        prediction = clf.predict(hog_features)

        if prediction == 1:
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
