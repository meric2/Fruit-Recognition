import os
import numpy as np
import pandas as pd
import tensorflow.keras.applications.vgg16 as vgg16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

def cnn_feature_extraction(resized_folder_path):
    # Load the VGG16 model with a smaller output layer
    model = vgg16.VGG16(weights="imagenet", include_top=False)
    # Get features from an earlier layer with fewer output units
    model = Model(inputs=model.input, outputs=model.get_layer(index=-4).output)

    CNN_features = []
    for root, dirs, files in os.walk(resized_folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                image_path = os.path.join(root, file)
                img = image.load_img(image_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x).flatten()  # Get features directly
                CNN_features.append(features)

    # Save features to a compressed CSV file
    df = pd.DataFrame(CNN_features)
    df.to_csv(resized_folder_path + "_CNN_features.csv.gz", compression="gzip")

# Call the function with the path to your resized images folder
resized_folder_path = "train"
cnn_feature_extraction(resized_folder_path)
print("CNN features extracted for training images")
resized_folder_path = "test"
cnn_feature_extraction(resized_folder_path)
print("CNN features extracted for test images")
resized_folder_path = "validation"
cnn_feature_extraction(resized_folder_path)
print("CNN features extracted for validation images")


