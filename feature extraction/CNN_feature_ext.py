import os
import numpy as np
import pandas as pd
import tensorflow.keras.applications.vgg16 as vgg16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm


def cnn_feature_extraction(resized_folder_path):
    # Load the VGG16 model without the top layer (fully connected layers)
    model = vgg16.VGG16(weights="imagenet", include_top=False)
    # Use an intermediate layer's output; adjust according to the desired depth
    model = Model(inputs=model.input, outputs=model.get_layer(index=-3).output)

    CNN_features = []
    for root, dirs, files in tqdm(os.walk(resized_folder_path)):
        for file in files:
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):  # Check for image files
                image_path = os.path.join(root, file)
                img = image.load_img(image_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x)

                # Determine the center of the feature map
                center_x, center_y = features.shape[1] // 2, features.shape[2] // 2

                start_x = max(center_x - 15, 0)
                start_y = max(center_y - 15, 0)
                end_x = min(center_x + 15, features.shape[1])
                end_y = min(center_y + 15, features.shape[2])

                mid_region_features = features[
                    0, start_x:end_x, start_y:end_y, :5  # Select the first 5 channels
                ].reshape(
                    -1
                )  # Flatten the region

                CNN_features.append(mid_region_features)

    # Save features to a compressed CSV file
    df = pd.DataFrame(CNN_features)
    df.to_csv(resized_folder_path + "_CNN_features.csv", index=False)


cnn_feature_extraction("data")
