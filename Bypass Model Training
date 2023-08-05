#Bypass model training, run GUI on seperate runtime and access previously trained saved model (weights and biases)

import numpy as np
import tensorflow as tf
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from facenet_pytorch import InceptionResnetV1
import numpy as np
import torch
import cv2
from albumentations import Compose, HorizontalFlip, Rotate, RandomBrightnessContrast, RandomScale

def images_to_embeddings(images, face_model, augment=True):
    embeddings = []
    
    # Define data augmentation pipeline
    if augment:
        augmentation_pipeline = Compose([
            HorizontalFlip(p=0.5),
            Rotate(limit=30, p=0.5),
            RandomBrightnessContrast(p=0.5),
            RandomScale(scale_limit=0.2, p=0.5)
        ])
    
    for count, image in enumerate(images):
        # Apply data augmentation if enabled
        if augment:
            augmented_image = augmentation_pipeline(image=image)
            image = augmented_image['image']
        
        image_rgb = np.stack([image] * 3, axis=-1)
        image_tensor = torch.tensor(image_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
        
        with torch.no_grad():
            embedding = face_model(image_tensor).numpy()
        
        embeddings.append(embedding)
        
        if (count + 1) % 50 == 0:
            print(f"Processed image {count + 1}/{len(images)}")
            
    return np.vstack(embeddings)
