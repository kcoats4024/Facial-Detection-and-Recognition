!pip install --user albumentations

#Pre-Trained Model v2

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

# Load LFW dataset
lfw_dataset = fetch_lfw_people(min_faces_per_person=20, resize=1)
X, y = lfw_dataset.images, lfw_dataset.target

# Preprocess the data
X = X / 255.0  # Normalize
y = LabelBinarizer().fit_transform(y)  # One-hot encoding

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Add channel dimension to the images
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# Load the pre-trained FaceNet model
face_model = InceptionResnetV1(pretrained='vggface2').eval()

# Define a function to convert images to FaceNet embeddings
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

# Convert the images to FaceNet embeddings
X_train_embeddings = images_to_embeddings(X_train[..., 0], face_model)
X_val_embeddings = images_to_embeddings(X_val[..., 0], face_model)

# Modify the model architecture to accept the embeddings
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_embeddings.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(lfw_dataset.target_names), activation='softmax')
])

# Compile the model with a smaller learning rate
optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the modified model
batch_size = 32
history = model.fit(
    X_train_embeddings, y_train,
    batch_size=batch_size,
    epochs=10000,
    validation_data=(X_val_embeddings, y_val),
    callbacks=[early_stopping]
)

# Plot the training and validation accuracy
print("Plotting Training and Validation Accuracy")
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
print("Plotting Training and Validation Loss")
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained model
print("Saving Model")
current_model = 'modelv6-Pretrainted-DataAugmentation-20IPF.h5' #"modelv6-Pretrainted-DataAugmentation-5FPI.h5", 'modelv6-Pretrainted-DataAugmentation-20IPF.h5
model.save(current_model)

# Predict on the validation set
y_val_pred = model.predict(X_val_embeddings)
y_val_pred_labels = np.argmax(y_val_pred, axis=1)
y_val_true_labels = np.argmax(y_val, axis=1)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_val_true_labels, y_val_pred_labels)
conf_mat = confusion_matrix(y_val_true_labels, y_val_pred_labels)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", conf_mat)

def get_unique_random_indices(true_labels, num_images):
    unique_indices = []
    unique_labels = set()
    
    while len(unique_indices) < num_images:
        random_index = np.random.randint(0, len(true_labels))
        random_label = true_labels[random_index]
        
        if random_label not in unique_labels:
            unique_indices.append(random_index)
            unique_labels.add(random_label)
    
    return unique_indices

# Display random images with unique true names
num_images_to_display = len(lfw_dataset.target_names)
unique_random_indices = get_unique_random_indices(y_val_true_labels, num_images_to_display)

for i in unique_random_indices:
    plt.imshow(X_val[i, :, :, 0], cmap='gray')
    plt.title(f"Predicted: {lfw_dataset.target_names[y_val_pred_labels[i]]}\nTrue: {lfw_dataset.target_names[y_val_true_labels[i]]}")
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

