import os

base_dir = ""


train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
valid_dir = os.path.join(base_dir, "valid")

num_of_classes = len([subdir for subdir in os.listdir(train_dir)])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
)

validatation_generator = validation_datagen.flow_from_directory(
    valid_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
)

# Construct the model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(150, 150, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(num_of_classes, activation="softmax"),
    ]
)


from tensorflow.keras.optimizers import RMSprop

model.compile(
    loss="categorical_crossentropy",
    optimizer=RMSprop(lr=0.001),
    metrics=["accuracy"],
)

history = model.fit(
    train_generator,
    steps_per_epoch=200,
    epochs=20,
    validation_data=validatation_generator,
    validation_steps=50,
    verbose=1,
)
