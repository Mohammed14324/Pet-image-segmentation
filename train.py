import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from functions import *   
import os
import warnings
warnings.filterwarnings("ignore")


train_dataset = get_dataset("data/train/images", "data/train/masks")

def Unet(input_shape):
    Input_layer = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(Input_layer)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    a5 = layers.Conv2D(256, 3, activation='relu', padding='same')(p4)
    a5 = layers.Conv2D(256, 3, activation='relu', padding='same')(a5)

    # Decoder
    d1 = layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='relu')(a5)
    d1 = layers.concatenate([d1, c4])
    d1 = layers.Conv2D(128, 3, activation='relu', padding='same')(d1)
    d1 = layers.Conv2D(128, 3, activation='relu', padding='same')(d1)

    d2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same', activation='relu')(d1)
    d2 = layers.concatenate([d2, c3])
    d2 = layers.Conv2D(64, 3, activation='relu', padding='same')(d2)
    d2 = layers.Conv2D(64, 3, activation='relu', padding='same')(d2)

    d3 = layers.Conv2DTranspose(32, 2, strides=2, padding='same', activation='relu')(d2)
    d3 = layers.concatenate([d3, c2])
    d3 = layers.Conv2D(32, 3, activation='relu', padding='same')(d3)
    d3 = layers.Conv2D(32, 3, activation='relu', padding='same')(d3)

    d4 = layers.Conv2DTranspose(16, 2, strides=2, padding='same', activation='relu')(d3)
    d4 = layers.concatenate([d4, c1])
    d4 = layers.Conv2D(16, 3, activation='relu', padding='same')(d4)
    d4 = layers.Conv2D(16, 3, activation='relu', padding='same')(d4)

    output_layer = layers.Conv2D(3, 1, activation="softmax")(d4)

    return models.Model(Input_layer, output_layer)

# Create model
model = Unet((128, 128, 3))

model.compile(
    optimizer="adam",
    loss=combined_loss,
    metrics=["accuracy"]
)

model.fit(
    train_dataset,
    batch_size=64,
    epochs=100,
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")

plt.show()

model.save("model/pet_segmentation.keras")