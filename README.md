# deevio_anomaly_detection
This project investigates the performance of unsupervised and supervised machine learning techniques for quality control using machine vision.

# Overview
This repo contains all the code used for this project. 
The supervised gradient boosting model used for image classification can be served via a REST API by running:

python deevio_app.py

The REST API requires you to submit a POST request with the image that you want to classify as a 'bytes' object.

The REST API can also be containerized before the endpoint is created using the Dockerfile.

The unsupervised model was only meant for testing purposed and cannot be served as a REST API without modification, since it did not perform as well as the supervised model.

# Report
A detailed analytical report discussing the methods and results can be found in Anomaly detection for automated quality control.pdf. 

# Dependencies
Use pip to install any missing dependencies found in requirements.txt.
pip3 install -r requirements.txt







 
