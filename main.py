#!/usr/bin/env python3

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Hyperparameters
activation_fn = "relu"
output_activation_fn = "softmax"
loss_fn = "categorical_crossentropy"
optimizer = "adam"
kernel_size = (3, 3)
pool_size = (2, 2)
batch_size = 128
epochs = 15

num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(f"x_train shape: {x_train.shape}")
print(f"{x_train.shape[0]} train samples")
print(f"{x_test.shape[0]} test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
	[
		keras.Input(shape=input_shape),
		layers.Conv2D(32, kernel_size=(3, 3), activation=activation_fn),
		layers.MaxPooling2D(pool_size=(2, 2)),
		layers.Conv2D(64, kernel_size=(3, 3), activation=activation_fn),
		layers.MaxPooling2D(pool_size=(2, 2)),
		layers.Flatten(),
		layers.Dropout(0.5),
		layers.Dense(num_classes, activation=output_activation_fn)
	]
)

model.summary()
model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")
