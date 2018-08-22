import requests
import deevio_app
from PIL import Image 
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import cv2

dirname = '/home/philip/Desktop/deevio_case_study/nailgun/nailgun/good'
filenames = [os.path.join(dirname + '_resized', fname)
                 for fname in os.listdir(dirname + '_resized')]

original_image = Image.open(filenames[0])

print(filenames[0])

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://127.0.0.1:5000/predict"
IMAGE_PATH = filenames[0] 

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# decode response
print("response\n", r["success"])

# ensure the request was successful
if r["success"]:
    print(r["predictions"])
    
# otherwise, the request failed
else:
    print("Request failed")
