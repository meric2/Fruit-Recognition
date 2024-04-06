"""
This code tests the BoVW model on a single image.
The user is prompted to enter the path of an image, and the model will predict the label of the image.

"""


import cv2
import numpy as np
import os
import random
#from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_model():
    # Assuming models are in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    svc_path = os.path.join(script_dir, 'svc_w_sift.pkl')
    kdtree_path = os.path.join(script_dir, 'kdtree.pkl')

    clf_loaded = joblib.load(svc_path)
    kdtree_loaded = joblib.load(kdtree_path)

    return clf_loaded, kdtree_loaded

def get_label_from_path(image_path):
    # Pick the folder name as the label
    parts = image_path.split(os.sep)
    return parts[-2]

def process_single_image(image_path, sift, kdtree, clf, n_clusters):
    # Extract SIFT features from a single image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    keypoints, descriptors = sift.detectAndCompute(img, None)

    # Generate BOVW histogram for the single image
    if descriptors is not None:
        histogram = kdtree.predict(descriptors)
        histogram = np.bincount(histogram, minlength=n_clusters)
    else:
        histogram = np.zeros(n_clusters)

    # Predict label for the single image
    prediction = clf.predict([histogram])
    
    return prediction, img

def main():
    # Load .pkl models
    clf_loaded, kdtree_loaded = load_model()

    # Create SIFT object with the optimized parameters
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10, sigma=1.2)

    while True:
        image_path = input("Enter the path of the image (or type 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            break

        true_label = get_label_from_path(image_path)
        prediction, img = process_single_image(image_path, sift, kdtree_loaded, clf_loaded, 30)

        if img is not None:
            print(f"True Label: {true_label}, Predicted Label: {prediction[0]}")
            cv2.putText(img, f"True: {true_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Pred: {prediction[0]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Image", img)
            cv2.waitKey(0)  # Waits indefinitely until a key is pressed
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()