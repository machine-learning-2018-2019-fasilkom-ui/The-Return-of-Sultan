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

percentage = 80
partition = int(len(hog_features)*percentage/100)
print("Partition:"+str(partition))

# Split train test dataset
x_train, x_test = data_frame[:partition, :-1],  data_frame[partition:, :-1]
y_train, y_test = data_frame[:partition, -  
                             1:].ravel(), data_frame[partition:, -1:].ravel()

clf.fit(x_train, y_train.ravel())
print("Finish Training")

y_pred = clf.predict(x_test)
y_pred_training = clf.predict(x_train)
print("Finish Predict")

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print("Accuracy Training: "+str(accuracy_score(y_train, y_pred_training)))
print('\n')
print(classification_report(y_test, y_pred))
