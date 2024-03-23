from feature_extraction import (
    extract_shape_features,
    calculate_texture_features,
    calculate_lbp_histogram,
    calculate_dominant_color,
)
import cv2
import os
import random
import numpy as np
import pandas as pd


def load_images(directory="data", batch_size=32):
    batch = []
    labels = []
    paths = []
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        image_names = os.listdir(folder_path)
        for i in range(2):
            image_name = random.choice(image_names)
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            batch.append(image)
            labels.append(folder_name)
            paths.append(image_path)
            if len(batch) == batch_size:
                yield batch
                batch = []
    return (batch, labels, paths)


def process_batch(batch):
    features = []
    for image in batch:
        shape_feature = extract_shape_features(image)
        texture_feature = calculate_texture_features(image)
        histogram = calculate_lbp_histogram(image)
        dominant_color = calculate_dominant_color(image)
        features.append((shape_feature, texture_feature, histogram, dominant_color))
    return features


def save_features(features, labels, prefix):
    shape_features = []
    texture_features = []
    histograms = []
    dominant_colors = []
    for shape_feature, texture_feature, histogram, dominant_color in features:
        shape_features.append(shape_feature)
        texture_features.append(texture_feature)
        histograms.append(histogram)
        dominant_colors.append(dominant_color)
    shapes = pd.DataFrame(
        shape_features,
        columns=["Area", "Perimeter", "Aspect Ratio", "Extent", "Solidity"],
    )
    texture = pd.DataFrame(texture_features, columns=["mean_lbp", "var_lbp"])
    histogram = pd.DataFrame(histograms)
    dominant_color = pd.DataFrame(dominant_colors, columns=["H", "S", "V"])
    batch_df = pd.concat([texture, dominant_color, histogram, shapes, labels], axis=1)
    batch_df.to_csv(f"{prefix}_features.csv", index=False)


def run_pipeline(directory):
    batch, labels, paths = load_images(directory)
    features = process_batch(batch)
    save_features(features, labels, "batch")
    paths_df = pd.DataFrame([paths, labels], columns=["path", "label"])
    paths_df.to_csv("paths.csv", index=False)


if __name__ == "__main__":
    run_pipeline("data")
