from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import hdbscan
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
    
from PIL import Image
import deevio_main

dirname_good = '/home/philip/Desktop/deevio_case_study/nailgun/nailgun/good'
dirname_bad = '/home/philip/Desktop/deevio_case_study/nailgun/nailgun/bad'

def ETL(dirname_good, dirname_bad, width=300):
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

    x_train, x_test, y_train, y_test = imgs[labels == 1], imgs, labels[labels == 1], labels
    image_size = x_train.shape[1]
    original_dim = image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test, y_train, y_test, original_dim


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae"):
    """Plots 2D latent space with labels overlayed.
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)
    
    filename = os.path.join(model_name, "vae_mean.png")
    
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    print(z_mean[:, 0])
    print(z_mean[:, 0].shape)
    print(type(z_mean[:, 0]))
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

x_train, x_test, y_train, y_test, original_dim = ETL(dirname_good, dirname_bad)


# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 10
latent_dim = 2
epochs = 10

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')
    
    plot_results(models,
                 data,
                 batch_size=batch_size,
model_name="vae_mlp")
    

    ### outlier score using dbscan 
    X = pd.DataFrame([])
    L = encoder.predict(x_test, batch_size=batch_size)
    
    enc_cns = ['latent_{}'.format(dim) for dim in range(L[0].shape[1])]
    for dim, cn in enumerate(enc_cns):
        print(dim, cn)
        X[cn] = L[0][:, dim]


    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, prediction_data=True).fit(L[0])
    X['anomaly_score'] = clusterer.outlier_scores_
    X['cluster'] = clusterer.labels_
    X['probabilities'] = clusterer.probabilities_
    X['label'] = y_test
    
    classification_report = classification_report(y_test, clusterer.labels_)
    
    print(X.head(50))   
    print("VAE classification_report: ", classification_report)

