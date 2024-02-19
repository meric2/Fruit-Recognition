import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt


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
        if descriptors is not None:
            histogram = np.zeros(len(kmeans.cluster_centers_))
            labels = kmeans.predict(descriptors)
            for label in labels:
                histogram[label] += 1
            histograms.append(histogram)
    return np.array(histograms)

def load_dataset(root_dir): # takes the folder name (class name) as label
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
    # Accuracy
    accuracy = accuracy_score(y, y_pred)

    # Match Accuracy
    match_accuracy = np.mean(y_pred == y)

    # Matching Performance
    matching_performance = np.sum(y_pred == y) / len(y)

    # Matching Precision and Recall
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')

    # Feature Count
    feature_count = X.shape[1]

    # Unique Match Ratio
    unique_match_ratio = len(np.unique(y_pred)) / len(np.unique(y))

    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred)

    return {
        'Accuracy': accuracy,
        'Match Accuracy': match_accuracy,
        'Matching Performance': matching_performance,
        'Precision': precision,
        'Recall': recall,
        'Feature Count': feature_count,
        'Unique Match Ratio': unique_match_ratio,
        'Precision-Recall Curve': (precision_curve, recall_curve)
    }

def display_results(image_paths, predicted_labels, true_labels, label_to_id):
    num_images = len(image_paths)
    num_cols = 4
    num_rows = num_images // num_cols + (1 if num_images % num_cols != 0 else 0)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    for idx, path in enumerate(image_paths):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(f"Predicted: {label_to_id[predicted_labels[idx]]}, True: {label_to_id[true_labels[idx]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Parameters
num_clusters = 100
root_dir_train = 'data/train'
root_dir_test = 'data/test'
root_dir_val = 'data/validation'

# Load datasets
image_paths_train, labels_train, label_to_id_train = load_dataset(root_dir_train)
image_paths_test, labels_test, label_to_id_test = load_dataset(root_dir_test)
image_paths_val, labels_val, label_to_id_val = load_dataset(root_dir_val)

# SIFT feature extractor and KMeans clustering model creation
sift = cv2.SIFT_create()
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

# Model creation and training
clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))
clf.fit(X_train, y_train)

# Model evaluation on test set
y_pred = clf.predict(X_test)
print("Model Evaluation on Test Set")
test_evaluation = model_evaluation(X_test, y_test, y_pred)
print(test_evaluation = model_evaluation(X_test, y_test, y_pred))

# Display results for the test set
display_results(image_paths_test, y_pred, y_val, label_to_id_test)

# Model evaluation on validation set
y_pred_val = clf.predict(X_val)
print("Model Evaluation on Validation Set")
validation_evaluation = model_evaluation(X_val, y_val, y_pred_val)
print(validation_evaluation)

# Display results for the validation set
display_results(image_paths_val, y_pred_val, y_val, label_to_id_val)