# https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

train_dir = "train"
validation_dir = "valid"

batch_size = 32
IMG_SIZE = (150, 150)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=IMG_SIZE,
    label_mode="categorical",
)

validation_dateset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=IMG_SIZE,
    label_mode="categorical",
)


class_names = train_dataset.class_names

IMG_SIZE = IMG_SIZE + (3,)  # add the channel dimension

base_model = InceptionV3(input_shape=IMG_SIZE, include_top=False, weights="imagenet")

base_model.trainable = False  # freeze the base model for transfer learning

num_classes = len(class_names)

base_model.summary()  # print the summary of the base model

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.20)(x)
x = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(
    optimizer=RMSprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]
)


# Add our data-augmentation parameters to ImageDataGenerator

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.90:
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=200,
    epochs=200,
    validation_steps=40,
    verbose=1,
    callbacks=[callbacks],
)


# Plot the training and validation loss and accuracy
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

plt.plot(epochs, acc, "r", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.show()

model.save("inceptionNet.h5")
