import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import SGD

INPUT_SHAPE = (150, 150)
batch_size = 32

base_dir = ""  # Base directory path

# Training and validation directories
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

# ImageDataGenerator objects
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

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=INPUT_SHAPE,
    batch_size=batch_size,  # Update this to use the batch_size variable
    class_mode="categorical",
    shuffle=True,
)

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=INPUT_SHAPE,
    batch_size=batch_size,  # Update this to use the batch_size variable
    class_mode="categorical",
)

# Model construction
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE + (3,)),
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
        layers.Dense(
            train_generator.num_classes, activation="softmax"
        ),  # Update this to use the dynamic number of classes
    ]
)

optimizer = SGD(learning_rate=0.001, momentum=0.9)


# Callback class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

# Model compilation
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # Corrected steps_per_epoch
    epochs=200,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples
    // batch_size,  # Consistency in validation steps
    verbose=1,
    callbacks=[callbacks],
)

# Save the model
model.save("CNN_model.h5")

# Plotting
import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, "bo", label="Training Accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.savefig("training_validation_accuracy.png")
