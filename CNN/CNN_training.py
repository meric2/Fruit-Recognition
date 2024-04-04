import os

INPUT_SHAPE = (150, 150)

base_dir = ""

train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

class_names = os.listdir(train_dir)

num_of_classes = len([subdir for subdir in os.listdir(train_dir)])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import resize_with_crop_or_pad
import tensorflow as tf

# ImageDataGenerator nesnelerini güncelleme
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=INPUT_SHAPE,
    batch_size=50,
    class_mode="categorical",
    shuffle=True,
)

validation_generator = validation_datagen.flow_from_directory(
    valid_dir, target_size=INPUT_SHAPE, batch_size=30, class_mode="categorical"
)

# Construct the model
import tensorflow as tf
from tensorflow.keras import models, layers

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=INPUT_SHAPE + (3,)
        ),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dense(num_of_classes, activation="softmax"),
    ]
)


from tensorflow.keras.optimizers import RMSprop, Adam, SGD

optimizer = SGD(learning_rate=0.001, momentum=0.9)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)

history = model.fit(
    train_generator,
    steps_per_epoch=150,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=1,
    callbacks=[callbacks],
)

model.save("CNN_model.h5")

import matplotlib.pyplot as plt

# Assuming 'history' is the variable containing the training history
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

# Plotting training & validation accuracy
# Adjusted plotting to handle unequal length arrays
min_epochs = min(len(epochs), len(val_accuracy))  # Find the minimum length
plt.plot(epochs[:min_epochs], accuracy[:min_epochs], "bo", label="Training Accuracy")
plt.plot(
    epochs[:min_epochs], val_accuracy[:min_epochs], "b", label="Validation Accuracy"
)
plt.title("Training and Validation Accuracy")
plt.legend()

plt.savefig("training_validation_accuracy.png")
