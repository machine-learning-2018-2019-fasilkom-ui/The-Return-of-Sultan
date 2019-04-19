import numpy as np
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold


# Load dataset
f = open(r'input/shipsnet.json')
dataset = json.load(f)
f.close()

# Convert dataset to numpy array
data = np.array(dataset['data']).astype('uint8')
labels = np.array(dataset['labels']).reshape(len(dataset['labels']), 1)

img_length = 80
data = data.reshape(-1, 3, img_length, img_length).transpose([0, 2, 3, 1])

# Convert image data RGB to Grayscale
data_gray = [color.rgb2gray(i) for i in data]

ppc = 16
hog_images = []
hog_features = []
# Extract HOG Features
for image in data_gray:
    fd, hog_image = hog(image, orientations=16, pixels_per_cell=(
        ppc, ppc), cells_per_block=(4, 4), block_norm='L2', visualize=True)
    hog_images.append(hog_image)
    hog_features.append(fd)


# Initialize Decision Tree Classifier using default parameters
clf = RandomForestClassifier(n_estimators=1000, random_state=17)

hog_features = np.array(hog_features)
data_frame = np.hstack((hog_features, labels))
np.random.shuffle(data_frame)

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)

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
    clf.fit(x_train, y_train.ravel())
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
print("f_score_0", f_score_0)
print("f_score_1", f_score_1)
print("weighted_avg_f1_score", weighted_avg_f1_score)
print("Fold Average F-Score Class 0", sum(f_score_0)/len(f_score_0))
print("Fold Average F-Score Class 1", sum(f_score_1)/len(f_score_1))
print("Fold Average weighted_avg_f1_score", sum(f_score_1)/len(f_score_1))
