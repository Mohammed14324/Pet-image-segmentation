import tensorflow as tf
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import os
from functions import *
from inference import predict_img
import warnings
warnings.filterwarnings("ignore")

test_dataset = get_dataset("data/test/images", "data/test/masks")
test_images = []
test_masks = []

for img, mask in test_dataset:
    test_images.append(img.numpy()[0] if len(img.shape) == 4 else img.numpy())
    test_masks.append(mask.numpy()[0] if len(mask.shape) == 4 else mask.numpy())

print(f"Total test images: {len(test_images)}")

history=model = tf.keras.models.load_model(
    "model/pet_segmentation.keras",
    custom_objects={'combined_loss': combined_loss}
)

for test in range(0, 5):
    img = predict_img(test_images[test], model)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original")
    if test_images[test].max() <= 1.0:
        plt.imshow(test_images[test])
    else:
        plt.imshow(test_images[test].astype(np.uint8))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Mask")
    if len(test_masks[test].shape) == 3:
        plt.imshow(test_masks[test])
    else:
        plt.imshow(test_masks[test], cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()