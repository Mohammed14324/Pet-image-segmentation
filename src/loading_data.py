import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Load dataset
dataset, info = tfds.load(
    'oxford_iiit_pet:4.0.0',
    with_info=True,
    download=True,
    as_supervised=False
)

train_dataset = dataset['train']
test_dataset  = dataset['test']
IMG_SIZE = 128
def save_images_masks(data, folder):
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder, "masks"), exist_ok=True)

    for i, sample in enumerate(tfds.as_numpy(data)):
        image = tf.image.resize(sample["image"], (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.uint8).numpy()
        Image.fromarray(image).save(os.path.join(folder, "images", f"{i}.png"))

        mask = tf.image.resize(sample["segmentation_mask"], (IMG_SIZE, IMG_SIZE), method="nearest")
        mask = tf.cast(mask, tf.int32) - 1
        mask = mask.numpy().squeeze(-1).astype(np.uint8)
        Image.fromarray(mask).save(os.path.join(folder, "masks", f"{i}.png"))

# Save train and test datasets
save_images_masks(train_dataset, "data/train")
save_images_masks(test_dataset, "data/test")
