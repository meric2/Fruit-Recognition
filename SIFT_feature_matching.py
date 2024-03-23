"""
Yontem - 1
Feature extraction using SIFT and Bag of Visual Words (BOVW) model
Feature extraction is done by SIFT.
Bag of Visual Words (BOVW) model is used to represent images as histograms of visual words.
Extracted features are then used to train a Support Vector Machine (SVM) model for image classification.
"""

import cv2
import numpy as np
import os
import random
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# Functions
def extract_features(image_paths, extractor):
    features = []
    for path in image_paths:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = extractor.detectAndCompute(gray, None)
        if descriptors is not None:
            features.append(descriptors)
    return np.concatenate(features, axis=0)


def create_bovw_histograms(image_paths, extractor, kmeans):
    histograms = []
    for path in image_paths:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = extractor.detectAndCompute(gray, None)
        histogram = np.zeros(
            len(kmeans.cluster_centers_)
        )  # Initialize histogram for each image
        if descriptors is not None:
            labels = kmeans.predict(descriptors)
            for label in labels:
                histogram[label] += 1
        histograms.append(histogram)
    return np.array(histograms)


def load_dataset(root_dir):  # takes the folder name (class name) as label
    image_paths = []
    labels = []
    label_to_id = {}
    for idx, class_name in enumerate(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            label_to_id[class_name] = idx
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                image_paths.append(file_path)
                labels.append(idx)
    return image_paths, labels, label_to_id


def model_evaluation(X, y, y_pred):
    # Match Accuracy
    match_accuracy = np.mean(y_pred == y)
    print("Match Accuracy:", match_accuracy)

    # Matching Precision and Recall
    precision = precision_score(y, y_pred, average="macro")
    print("Precision:", precision)
    recall = recall_score(y, y_pred, average="macro")
    print("Recall:", recall)

    # F1 Score
    f1 = f1_score(y, y_pred, average="macro")
    print("F1 Score:", f1)

    # Feature Count
    feature_count = X.shape[1]
    print("Feature Count:", feature_count)

    # Unique Match Ratio
    unique_match_ratio = len(np.unique(y_pred)) / len(np.unique(y))
    print("Unique Match Ratio:", unique_match_ratio)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(10, 7)) # Adjust the size as needed
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        "Match Accuracy": match_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Feature Count": feature_count,
        "Unique Match Ratio": unique_match_ratio,
        "Confusion Matrix": cm
    }


def display_all_results(image_paths, predicted_labels, true_labels, label_to_id):
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_images = len(image_paths)
    num_cols = 4
    num_rows = num_images // num_cols + (1 if num_images % num_cols != 0 else 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    for idx, path in enumerate(image_paths):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        pred_label = id_to_label.get(predicted_labels[idx], "Unknown")
        true_label = id_to_label.get(true_labels[idx], "Unknown")
        result = "TRUE" if predicted_labels[idx] == true_labels[idx] else "FALSE"
        ax.set_title(f"Predicted: {pred_label}, True: {true_label}\n{result}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def display_sample_results(
    image_paths, predicted_labels, true_labels, label_to_id, sample_size=10
):
    id_to_label = {v: k for k, v in label_to_id.items()}
    sample_size = min(sample_size, len(image_paths))

    # Randomly sample indices without replacement
    sampled_indices = random.sample(range(len(image_paths)), sample_size)

    # Subset the data
    sampled_image_paths = [image_paths[i] for i in sampled_indices]
    sampled_predicted_labels = [predicted_labels[i] for i in sampled_indices]
    sampled_true_labels = [true_labels[i] for i in sampled_indices]

    num_cols = 4
    num_rows = sample_size // num_cols + (1 if sample_size % num_cols != 0 else 0)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(15, 5 * num_rows), squeeze=False
    )
    axes = axes.flatten()

    for idx, path in enumerate(sampled_image_paths):
        ax = axes[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        pred_label = id_to_label.get(sampled_predicted_labels[idx], "Unknown")
        true_label = id_to_label.get(sampled_true_labels[idx], "Unknown")
        result = (
            "TRUE"
            if sampled_predicted_labels[idx] == sampled_true_labels[idx]
            else "FALSE"
        )
        ax.set_title(f"Predicted: {pred_label}, True: {true_label}\n{result}")
        ax.axis("off")

    for idx in range(len(sampled_image_paths), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


# Parameters
num_clusters = 30
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
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features_train)

# Create BOVW histograms for train, test, and validation sets
bovw_histograms_train = create_bovw_histograms(image_paths_train, sift, kmeans)
bovw_histograms_test = create_bovw_histograms(image_paths_test, sift, kmeans)
bovw_histograms_val = create_bovw_histograms(image_paths_val, sift, kmeans)

# Train, test, and validation data split
X_train, y_train = bovw_histograms_train, labels_train
X_test, y_test = bovw_histograms_test, labels_test
X_val, y_val = bovw_histograms_val, labels_val

# Model creation and training # Hyperparameter tuning for SVC is done on y1_para_opt.ipynb
clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", kernel="rbf"))
clf.fit(X_train, y_train)

# Model evaluation on test set
y_pred = clf.predict(X_test)
print("Model Evaluation on Test Set")
test_evaluation = model_evaluation(X_test, y_test, y_pred)

# Display results for the test set
# display_all_results(image_paths_test, y_pred, y_test, label_to_id_test)
display_sample_results(image_paths_test, y_pred, y_test, label_to_id_test)


# Model evaluation on validation set
y_pred_val = clf.predict(X_val)
print("Model Evaluation on Validation Set")
validation_evaluation = model_evaluation(X_val, y_val, y_pred_val)

# Display results for the validation set
# display_all_results(image_paths_val, y_pred_val, y_val, label_to_id_val)
display_sample_results(image_paths_val, y_pred_val, y_val, label_to_id_val)
