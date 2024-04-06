"""
Makes predictions on a single image using a trained xgb10_model.pkl
"""

import os
import cv2
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from feature_extraction import (
    extract_shape_features,
    calculate_lbp_histogram,
    calculate_texture_features,
    calculate_dominant_color,
)


# Function to combine all feature extraction methods
def extract_features(image):
    # Extract shape features from the image and store in the input_array
    shape_feature = extract_shape_features(image)

    texture_feature = calculate_texture_features(image)
    histogram = [0, 0]
    color = calculate_dominant_color(image)

    return_vector = np.concatenate(
        (shape_feature, texture_feature, histogram, color), axis=None
    )

    return return_vector


def predict_image(image_path, model, scaler, label_encoder):
    # Load the image
    image = cv2.imread(image_path)

    # Extract features from the image
    input_vector = extract_features(image)

    input_vector = input_vector.reshape(1, -1)

    # Scale the features using the provided scaler
    scaled_features = scaler.transform(input_vector)

    # Predict using the trained model
    prediction = model.predict(scaled_features)

    # Decode the prediction to get the label
    label = label_encoder.inverse_transform([prediction[0]])[0]

    return label


def main():
    # Load the trained model and other necessary components
    model = joblib.load("xgb10_model.pkl")
    scaler = joblib.load(
        "xgb10_scaler.pkl"
    )  # Ensure you have saved the scaler during training
    label_encoder = joblib.load(
        "xgb10_label_encoder.pkl"
    )  # Ensure you have saved the label encoder during training

    while True:
        image_path = input("Enter the path of the image ('exit' to quit): ")
        if image_path.lower() == "exit":
            print("Exiting program.")
            break
        if not os.path.isfile(image_path):
            print(f"No file found at {image_path}, please try again.")
            continue

        # Predict and print the result
        label = predict_image(image_path, model, scaler, label_encoder)
        print(f"The image is predicted to be: {label}")


if __name__ == "__main__":
    main()
