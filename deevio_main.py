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
from sklearn import datasets, svm, metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
    
def scale_image(input_image_path, output_image_path, width=None, height=None):
    """Loads the images from a directory.
       Saves a scaled version of the original image.
    """

    original_image = Image.open(input_image_path)
    w, h = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=w, height=h))
 
    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')
 
    original_image.thumbnail(max_size, Image.ANTIALIAS)
    t = len(input_image_path) - (len(output_image_path) - 8)
    original_image.save(output_image_path + input_image_path[-t:])
 
    scaled_image = Image.open(output_image_path + input_image_path[-t:])
    width, height = scaled_image.size
    print('The scaled image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))

def load_data(dirname, width):
    """Loads the images from a directory.
       Returns images and labels as numpy arrays.
    """

    # Load every image file in the provided directory
    filenames = [os.path.join(dirname, fname)
                 for fname in os.listdir(dirname)]
    
    [scale_image(fname, output_image_path = dirname + '_resized/', width=width) for fname in filenames]
   

    filenames = [os.path.join(dirname + '_resized', fname)
                 for fname in os.listdir(dirname + '_resized')]

    # Read every filename
    imgs = np.array([plt.imread(fname) for fname in filenames])
    labels = [fname.split('_')[-1][:-5] for fname in filenames]
    images_and_labels = list(zip(imgs, labels))

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(imgs)
    
    nsamples, nx, ny = imgs.shape
    imgs = imgs.reshape((nsamples,nx*ny))
    print(n_samples, nsamples)

    return imgs, labels
