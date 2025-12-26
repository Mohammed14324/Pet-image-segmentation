import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.functions import combined_loss
import warnings
warnings.filterwarnings("ignore")

def predict_img(img, model):
    if isinstance(img, tf.Tensor):
        img = img.numpy()
    
    if isinstance(img, np.ndarray):
        while len(img.shape) > 3 and img.shape[0] == 1:
            img = img[0]
        
        if len(img.shape) == 4:
            img = img[0]
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        img = Image.fromarray(img)
    
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    pred_array = np.clip(prediction[0] * 255.0, 0, 255).astype(np.uint8)
    
    result_img = (img_array[0] * 255.0).astype(np.uint8)
    
    pred_classes = np.argmax(pred_array, axis=-1)
    mask = (pred_classes != 1)
    
    mask_3d = np.expand_dims(mask, axis=-1)
    result_img = np.where(mask_3d, 255 - result_img, result_img)
    
    pred_img = Image.fromarray(result_img)
    return pred_img