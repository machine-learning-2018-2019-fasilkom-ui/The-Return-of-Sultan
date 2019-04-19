import numpy as np  # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
import pickle
from sklearn import tree
from math import log, sqrt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class CART(object):
    def __init__(self, tree='cls', criterion='gini', prune='depth', max_depth=4, min_criterion=0.05):
        self.feature = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0

        self.root = None
        self.criterion = criterion
        self.prune = prune
        self.max_depth = max_depth
        self.min_criterion = min_criterion
        self.tree = tree

    def fit(self, features, target):
        self.root = CART()
        if(self.tree == 'cls'):
            self.root._grow_tree(features, target, self.criterion)
        else:
            self.root._grow_tree(features, target, 'mse')
        self.root._prune(self.prune, self.max_depth,
                         self.min_criterion, self.root.n_samples)

    def predict(self, features):
        return np.array([self.root._predict(f) for f in features])

    def print_tree(self):
        self.root._show_tree(0, ' ')

    def _grow_tree(self, features, target, criterion='gini'):
        self.n_samples = features.shape[0]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        if criterion in {'gini', 'entropy'}:
            self.label = max([(c, len(target[target == c]))
                              for c in np.unique(target)], key=lambda x: x[1])[0]
        else:
            self.label = np.mean(target)
        print("_grow_tree(label):"+str(self.label))
        impurity_node = self._calc_impurity(criterion, target)
        print("_calc_impurity:"+str(impurity_node))
        print("Total Features:"+str(features.shape[1]))
        for col in range(features.shape[1]):
            feature_level = np.unique(features[:, col])
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            for threshold in thresholds:
                target_l = target[features[:, col] <= threshold]
                impurity_l = self._calc_impurity(criterion, target_l)
                n_l = float(target_l.shape[0]) / self.n_samples

                target_r = target[features[:, col] > threshold]
                impurity_r = self._calc_impurity(criterion, target_r)
                n_r = float(target_r.shape[0]) / self.n_samples

                impurity_gain = impurity_node - \
                    (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = col
                    best_threshold = threshold

        self.feature = best_feature
        self.gain = best_gain
        self.threshold = best_threshold
        self._split_tree(features, target, criterion)

    def _split_tree(self, features, target, criterion):
        features_l = features[features[:, self.feature] <= self.threshold]
        target_l = target[features[:, self.feature] <= self.threshold]
        self.left = CART()
        self.left.depth = self.depth + 1
        self.left._grow_tree(features_l, target_l, criterion)

        features_r = features[features[:, self.feature] > self.threshold]
        target_r = target[features[:, self.feature] > self.threshold]
        self.right = CART()
        self.right.depth = self.depth + 1
        self.right._grow_tree(features_r, target_r, criterion)

    def _calc_impurity(self, criterion, target):
        if criterion == 'gini':
            return 1.0 - sum([(float(len(target[target == c])) / float(target.shape[0])) ** 2.0 for c in np.unique(target)])
        elif criterion == 'mse':
            return np.mean((target - np.mean(target)) ** 2.0)
        else:
            entropy = 0.0
            for c in np.unique(target):
                p = float(len(target[target == c])) / target.shape[0]
                if p > 0.0:
                    entropy -= p * np.log2(p)
            return entropy

    def _prune(self, method, max_depth, min_criterion, n_samples):
        if self.feature is None:
            return

        self.left._prune(method, max_depth, min_criterion, n_samples)
        self.right._prune(method, max_depth, min_criterion, n_samples)

        pruning = False

        if method == 'impurity' and self.left.feature is None and self.right.feature is None:
            if (self.gain * float(self.n_samples) / n_samples) < min_criterion:
                pruning = True
        elif method == 'depth' and self.depth >= max_depth:
            pruning = True

        if pruning is True:
            self.left = None
            self.right = None
            self.feature = None

    def _predict(self, d):
        if self.feature is not None:
            if d[self.feature] <= self.threshold:
                return self.left._predict(d)
            else:
                return self.right._predict(d)
        else:
            return self.label

    def _show_tree(self, depth, cond):
        base = '    ' * depth + cond
        if self.feature is not None:
            print(base + 'if X[' + str(self.feature) +
                  '] <= ' + str(self.threshold))
            self.left._show_tree(depth+1, 'then ')
            self.right._show_tree(depth+1, 'else ')
        else:
            print(base + '{value: ' + str(self.label) +
                  ', samples: ' + str(self.n_samples) + '}')


def accuracy(pred, true):
    correct = 0
    pred_len = len(pred)
    for i in range(pred_len):
        if pred[i] == true[i]:
            correct += 1
    return correct/pred_len


def test_decision_tree():
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

    hog_features = np.array(hog_features)
    print("hog_features[0]:"+str(hog_features.shape[0]))
    print("hog_features[1]:"+str(hog_features.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(
        hog_features, labels, test_size=0.3, random_state=17)

    print("Starting DT.fit")
    dt = CART(tree='cls', criterion='entropy', prune='impurity', max_depth=3)
    dt.fit(X_train, y_train)
    print("Finished DT.fit")
    y_pred = dt.predict(X_test)
    print("Prediction:"+str(y_pred))
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:"+str(acc))
    print('\n')
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    test_decision_tree()
