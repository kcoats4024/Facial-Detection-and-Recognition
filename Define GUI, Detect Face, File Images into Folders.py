#GUI
#Zoomed face, files into folders

current_model = 'modelv6-Pretrainted-DataAugmentation-20IPF.h5'
import urllib.request
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from skimage.transform import resize
from facenet_pytorch import InceptionResnetV1

def download_deploy_prototxt():
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    urllib.request.urlretrieve(url, "deploy.prototxt")

# download the file
download_deploy_prototxt()

def download_caffe_model():
    url = "https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel?raw=true"
    urllib.request.urlretrieve(url, "res10_300x300_ssd_iter_140000.caffemodel")

# download the file
download_caffe_model()

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os
import shutil
from tensorflow.keras.models import load_model
import cv2
from sklearn.datasets import fetch_lfw_people

lfw_dataset = fetch_lfw_people(min_faces_per_person=20, resize=0.4)

def evaluate_image_with_model(img_path, model_path, target_shape, target_names):
    # Load the image and convert to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (target_shape[1], target_shape[0]))
    
    # Add the channel and batch dimensions
    input_image = img[..., np.newaxis][np.newaxis, ...]
    input_image = input_image / 255.0  # Normalize

    # Load the pre-trained FaceNet model
    face_model = InceptionResnetV1(pretrained='vggface2').eval()

    # Convert the image to FaceNet embedding
    input_image_embedding = images_to_embeddings(input_image[..., 0], face_model, augment=False)

    # Load the trained model
    loaded_model = tf.keras.models.load_model(model_path)

    # Perform the prediction using the embedding
    prediction = loaded_model.predict(input_image_embedding)
    predicted_class_index = np.argmax(prediction)
    predicted_class_probability = np.max(prediction)
    return target_names[predicted_class_index], predicted_class_probability


def load_and_preprocess_image(image_path, target_shape):
    input_image = image.load_img(image_path, color_mode='grayscale', target_size=target_shape)
    input_image = np.array(input_image) / 255.0  # Normalize
    input_image_final = np.expand_dims(input_image, axis=[0, -1])  # Add batch and channel dimensions
    return input_image_final, input_image

def browse_image():
    global img_path
    img_path = filedialog.askopenfilename()
    load_image(img_path)

def detect_and_crop_face(image_path, prototxt_path, model_path, conf_threshold=0.5):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            return face
    
def load_image(image_path):
    face_cascade_path = "deploy.prototxt"
    face_model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    face = detect_and_crop_face(image_path, face_cascade_path, face_model_path)
    if face is not None:
        img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        label_image.config(image=img)
        label_image.image = img
    else:
        result_label.config(text="No face detected")

def evaluate_button_clicked():
    global img_path, predicted_name
    model_path = current_model  
    target_shape = (125, 94, 1)
    target_names = lfw_dataset.target_names
    predicted_name, predicted_probability = evaluate_image_with_model(img_path, model_path, target_shape, target_names)
    result_label.config(text=f"Predicted: {predicted_name} with probability: {predicted_probability:.2f}")

def confirm_button_clicked():
    target_folder = os.path.join("target_folders", predicted_name)
    if img_path and os.path.exists(target_folder):
        shutil.move(img_path, os.path.join(target_folder, os.path.basename(img_path)))
        result_label.config(text=f"Moved to: {os.path.join(target_folder, os.path.basename(img_path))}")
    else:
        result_label.config(text="Error: Image or target folder not found")

def create_target_folders(target_names, base_folder="target_folders"):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    for target_name in target_names:
        target_folder = os.path.join(base_folder, target_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

def show_recognizable_names():
    recognizable_names_window = tk.Toplevel(root)
    recognizable_names_window.title("Recognizable Names")
    recognizable_names_window.geometry("300x400")
    
    names_label = tk.Label(recognizable_names_window, text="Names of people the model can recognize:")
    names_label.pack()
    
    scrollbar = tk.Scrollbar(recognizable_names_window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    names_listbox = tk.Listbox(recognizable_names_window, yscrollcommand=scrollbar.set)
    
    for name in target_names:
        names_listbox.insert(tk.END, name)
        
    names_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
    scrollbar.config(command=names_listbox.yview)            
            
# Create the main window
root = tk.Tk()
root.title("Image Classification")

# Create widgets
frame = tk.Frame(root)
frame.pack()

label_image = tk.Label(frame)
label_image.pack()

button_browse = tk.Button(frame, text="Browse Image", command=browse_image)
button_browse.pack()

button_evaluate = tk.Button(frame, text="Evaluate", command=evaluate_button_clicked)
button_evaluate.pack()

button_confirm = tk.Button(frame, text="Confirm", command=confirm_button_clicked)
button_confirm.pack()

button_show_names = tk.Button(frame, text="Show Recognizable Names", command=show_recognizable_names)
button_show_names.pack()

result_label = tk.Label(frame, text="")
result_label.pack()

# Call the functions to create target folders and download files
target_names = lfw_dataset.target_names
create_target_folders(target_names)
download_deploy_prototxt()
download_caffe_model()

# Start the main loop
root.mainloop()
