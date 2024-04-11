"""
Yontem - 1
Feature extraction using SIFT and Bag of Visual Words (BOVW) model
Feature extraction is done by SIFT.
Bag of Visual Words (BOVW) model is used to represent images as histograms of visual words.
Extracted features are then used to train a k-Nearest Neighbor (kNN) model for image classification.
This script can be used to test the models.

"""

import cv2
import numpy as np
import os
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from SIFT_feature_matching import (
    extract_features,
    create_bovw_histograms,
    load_dataset,
    model_evaluation,
    display_all_results,
    display_sample_results,
)
import warnings

warnings.filterwarnings("ignore")

# Load trained models
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "knn_sift.pkl")
clf_loaded = joblib.load(save_path)

save_path = os.path.join(script_dir, "kmeans.pkl")
kmeans_loaded = joblib.load(save_path)


# Parameters
num_clusters = 20
root_dir_train = "data_128x128/train"
root_dir_test = "data_128x128/test"
root_dir_val = "data_128x128/validation"

# Load datasets
image_paths_train, labels_train, label_to_id_train = load_dataset(root_dir_train)
image_paths_test, labels_test, label_to_id_test = load_dataset(root_dir_test)
image_paths_val, labels_val, label_to_id_val = load_dataset(root_dir_val)

# SIFT feature extractor and KMeans clustering model creation # SIFT parameters are tuned on y1_para_opt.ipynb
sift = cv2.SIFT_create(
    nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10, sigma=1.2
)
features_train = extract_features(image_paths_train, sift)

# Create BOVW histograms for train, test, and validation sets
bovw_histograms_train = create_bovw_histograms(image_paths_train, sift, kmeans_loaded)
bovw_histograms_test = create_bovw_histograms(image_paths_test, sift, kmeans_loaded)
bovw_histograms_val = create_bovw_histograms(image_paths_val, sift, kmeans_loaded)

# Train, test, and validation data split
X_train, y_train = bovw_histograms_train, labels_train
X_test, y_test = bovw_histograms_test, labels_test
X_val, y_val = bovw_histograms_val, labels_val


# Model evaluation on test set
y_pred = clf_loaded.predict(X_test)
print("Model Evaluation on Test Set")
test_evaluation = model_evaluation(X_test, y_test, y_pred)

# Display results for the test set
# display_all_results(image_paths_test, y_pred, y_test, label_to_id_test)
display_sample_results(image_paths_test, y_pred, y_test, label_to_id_test)


# Model evaluation on validation set
y_pred_val = clf_loaded.predict(X_val)
print("Model Evaluation on Validation Set")
validation_evaluation = model_evaluation(X_val, y_val, y_pred_val)

# Display results for the validation set
# display_all_results(image_paths_val, y_pred_val, y_val, label_to_id_val)
display_sample_results(image_paths_val, y_pred_val, y_val, label_to_id_val)
