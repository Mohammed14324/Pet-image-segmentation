import tensorflow as tf
import tensorflow.keras.backend as K
import os
import warnings
warnings.filterwarnings("ignore")

IMG_SIZE = 128
NUM_CLASSES = 3
BATCH_SIZE = 64
# Custom Loss: Dice Loss + Categorical Crossentropy
def combined_loss(y_true, y_pred, smooth=1e-6):
    # Flatten tensors for Dice calculation
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # Dice coefficient
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice_loss = 1 - dice
    
    # Crossentropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dice_loss + K.mean(cce)


def parse_image_mask(image_path, mask_path):
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0

    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method="nearest")
    mask = tf.cast(mask, tf.int32)
    mask = tf.one_hot(mask[..., 0], NUM_CLASSES)

    return image, mask


def get_dataset(image_dir, mask_dir):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_files  = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    
    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.map(parse_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset