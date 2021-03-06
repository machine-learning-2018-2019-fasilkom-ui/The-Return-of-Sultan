# This Python 3 environment comes with many helpful analytics libraries 
# installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/
# docker-python
# For example, here's several helpful packages to load in


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
from subprocess import call
import PIL
from matplotlib import patches
import os



import pickle
from gzip_pickle import *
from sklearn.model_selection import StratifiedKFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will
# list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

try:
    pickle_in = open("aug_dataset.pickle", "rb")
    pickle_in_labels = open("aug_dataset_labels.pickle", "rb")
    data = pickle.load(pickle_in)
    labels = pickle.load(pickle_in_labels)
    pickle_in.detach()
    pickle_in_labels.detach()
except:
    pickle_out = open("aug_dataset.pickle", "wb")
    pickle_out_labels = open("aug_dataset_labels.pickle", "wb")
    data = []
    labels = []
    path = "augmented_data/"
    valid_images = [".png"]

    for f in os.listdir(path):
        print("path folder:", f)
        ext = os.path.splitext(f)[1]
        path_class = path+f+'/'
        for img in os.listdir(path_class):
            # print("file name:", img)
            ext_file = os.path.splitext(img)[1]
            # print("file ext:", ext_file)
            if ext_file.lower() not in valid_images:
                continue
            temp = PIL.Image.open(os.path.join(path_class, img))
            data.append(np.array(temp.copy()).ravel())
            labels.append(int(f))
            temp.close()

    pickle.dump(data, pickle_out)
    pickle.dump(labels, pickle_out_labels)
    pickle_out.close()
    pickle_out_labels.close()

data = np.array(data).astype('uint8')
labels = np.array(labels).reshape(len(labels), 1)
print(data)
print(labels)


data.shape
img_length = 80
print(data.shape)
data = data.reshape(-1, img_length, img_length, 3).transpose([0, 1, 2, 3])
print(data.shape)



data_gray = [color.rgb2gray(i) for i in data]


ppc = 16
hog_images = []
hog_features = []
for image in data_gray:
    fd, hog_image = hog(image, orientations=16, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2', visualize=True)
    hog_images.append(hog_image)
    hog_features.append(fd)


clf = tree.DecisionTreeClassifier(random_state=17)
print(clf)


hog_features = np.array(hog_features)
data_frame = np.hstack((hog_features, labels))

# # Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    hog_features, labels.ravel(), test_size=0.2, random_state=17)

tuned_parameters = {"max_depth": [10, 20, 30, 40, 50],
                    "max_features": ["sqrt"],
                    "min_samples_leaf": [0.1, 0.2, 0.3, 0.4],
                    "min_samples_split": np.linspace(0.1, 1.0, 10,
                                                     endpoint=True),
                    "criterion": ["gini"],
                    "class_weight": ["balanced"]}

# Best param ROC AUC
# {'min_samples_split': 0.2, 'min_samples_leaf': 0.1, 'max_features': 'log2', 
# 'max_depth': 12.0, 'criterion': 'gini', 'class_weight': 'balanced'}
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(20, 100, num=10)]
# max_depth = [18]
# max_depth.append(None)
# Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
min_samples_split = np.linspace(0.1, 0.5, 10, endpoint=True)
# min_samples_split = [0.1]
# Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# min_samples_leaf = [0.1]
# Method of selecting samples for training each tree
# bootstrap = [True, False]
bootstrap = [True]
class_weight = ["balanced"]

# Create the random grid
tuned_parameters = {'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'class_weight': class_weight}




scores = ['f1_macro']

best_estimator = []
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                       param_grid=tuned_parameters, cv=3,
                       scoring=score, n_jobs=-1,
                       verbose=2)
    clf.fit(X_train, y_train)

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

# Stratified K-Fold Cross Validation
# skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)
# skf.get_n_splits(hog_features, labels)
# print(skf)

X_train, X_test, y_train, y_test = train_test_split(
    hog_features, labels.ravel(), test_size=0.8, random_state=0)



clf = best_estimator[0]
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
