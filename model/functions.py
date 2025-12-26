# Custom Loss: Dice Loss + Categorical Crossentropy
def combined_loss(y_true, y_pred, smooth=1e-6):
    # Flatten tensors for Dice calculation
    import keras.backend as K
    import tensorflow as tf
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # Dice coefficient
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice_loss = 1 - dice
    
    # Crossentropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dice_loss + K.mean(cce)