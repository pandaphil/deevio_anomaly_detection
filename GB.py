import os
import numpy as np
import utils
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')

import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report

import deevio_main

dirname_good = '/home/philip/Desktop/deevio_case_study/nailgun/nailgun/good'
dirname_bad = '/home/philip/Desktop/deevio_case_study/nailgun/nailgun/bad'

def ETL(dirname_good, dirname_bad, width=1000):
    """Load and preprocess images.
    """
    print(">>> Loading and preprocessing")

    imgs_good, labels_good = deevio_main.load_data(dirname_good, width)
    imgs_bad, labels_bad = deevio_main.load_data(dirname_bad, width)

    imgs = np.array(list(imgs_good) + list(imgs_bad))
    labels = labels_good + labels_bad

    imgs = np.array(imgs)
    imgs, labels = shuffle(imgs, labels, random_state=0)
    print(labels)
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels) 
    print(labels)

    x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def train_test_model():
    """Training, testing and evaluating the model.
    """

    x_train, x_test, y_train, y_test = ETL(dirname_good, dirname_bad)

    # Create a classifier: gradient boosting classifier
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    print(">>> Training")
    classifier.fit(x_train, y_train)

    expected = y_test

    print(">>> Predicting")
    predicted = classifier.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    result = classifier.score(x_test, y_test)
    print(result)
    return classifier, predicted, result

def save_model():
    """Save the model to disk.
    """
    classifier, predicted, result = train_test_model()

    print(">>> Saving model")
    filename = './model/supervised_GB_model.pkl'
    pickle.dump(classifier, open(filename, 'wb'))

if __name__ == "__main__":
    train_test_model()
    #save_model()
