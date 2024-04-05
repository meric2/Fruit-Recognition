import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model("ResNet_model.h5")


def predict_image(image_path, model):
    # Load and prepare the image
    img = image.load_img(
        image_path, target_size=(160, 160)
    )  # Adjust the size according to your model's input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch.
    img_array = preprocess_input(img_array)  # Use ResNet50's preprocess_input

    # Make prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)

    # Assuming your model's classes are in the same order as the training
    classes = sorted(
        os.listdir("train")
    )  # Adjust this if the path or method to retrieve class names differs

    return classes[class_index[0]]


def main():
    while True:
        image_path = input("Enter the path of the image ('exit' to quit): ")
        if image_path.lower() == "exit":
            print("Exiting program.")
            break
        if not os.path.isfile(image_path):
            print(f"No file found at {image_path}, please try again.")
            continue

        # Predict and print the result
        label = predict_image(image_path, model)
        print(f"The image is predicted to be: {label}")


if __name__ == "__main__":
    main()
