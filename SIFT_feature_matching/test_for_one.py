"""
This code tests the BoVW model on a single image.
The user is prompted to enter the path of an image, and the model will predict the label of the image.

"""


import cv2
import numpy as np
import os
import random
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_model():
    # Assuming models are in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    svc_path = os.path.join(script_dir, 'svc_w_sift.pkl')
    kmeans_path = os.path.join(script_dir, 'kmeans.pkl')

    clf_loaded = joblib.load(svc_path)
    kmeans_loaded = joblib.load(kmeans_path)

    return clf_loaded, kmeans_loaded

def load_label_mapping(labels_path):
    with open(labels_path, 'r') as file:
        labels = file.read().splitlines()
    
    # Create a mapping from numeric labels to fruit names
    # Adjust the starting index based on your model's labeling convention (0 or 1)
    label_mapping = {i: label for i, label in enumerate(labels)}
    
    return label_mapping


def process_single_image(image_path, sift, kmeans, clf, label_mapping):
    # Extract SIFT features from a single image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    keypoints, descriptors = sift.detectAndCompute(img, None)

    # Generate BOVW histogram for the single image
    if descriptors is not None:
        histogram = kmeans.predict(descriptors)
        histogram = np.bincount(histogram, minlength=kmeans.n_clusters)
    else:
        histogram = np.zeros(kmeans.n_clusters)

    # Predict label for the single image
    prediction = clf.predict([histogram])
    prediction_label = label_mapping.get(prediction[0], "Unknown")  # Convert numeric label to class name
    
    return prediction_label, img

def main():
    # Load .pkl models
    clf_loaded, kmeans_loaded = load_model()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    labels_path = os.path.join(script_dir, 'labels.txt')
    label_mapping = load_label_mapping(labels_path)

    # Create SIFT object with the optimized parameters
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10, sigma=1.2)

    while True:
        image_path = input("Enter the path of the image (or type 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            break

        prediction, img = process_single_image(image_path, sift, kmeans_loaded, clf_loaded)

       
        print(f"Predicted Label: {prediction[0]}")

if __name__ == "__main__":
    main()
