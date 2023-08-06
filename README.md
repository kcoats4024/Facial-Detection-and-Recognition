# FaceNet Embeddings with Dense Classifier: A Transfer Learning Approach for LFW Face Recognition
This repository contains a face recognition model built using a transfer learning approach. The model leverages embeddings from the pre-trained FaceNet model, followed by a dense neural network, to perform classification on the Labeled Faces in the Wild (LFW) dataset.
![Accuracy and Loss of Model](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/b453a31c-6590-4f48-b676-032816770375)

## Model Overview
FaceNet: The InceptionResnetV1 model from the facenet_pytorch library is utilized to convert face images into embeddings.
Dense Neural Network: Once the embeddings are obtained, a dense neural network classifier is trained on these embeddings to recognize different faces.
Key Features
Transfer Learning: Instead of training a deep model from scratch, the power of the pre-trained FaceNet model is harnessed to generate embeddings.
Data Augmentation: To enhance the training data and increase generalization, various data augmentation techniques, such as horizontal flip and rotation, are employed.
Regularization: The dense classifier utilizes dropout and L2 regularization to reduce the risk of overfitting.

## Graphical User Interface (GUI) Summary:
- Browse Image Button: Allows users to select an image file to be recognized. The model can identify if the face in the image is one of the recognizable faces.
- Evaluate Button: Once an image is selected, this button processes the image through the model. The prediction, along with a confidence score, is then displayed.
- Confirm Button: Once a prediction is made, users can confirm its accuracy. The image then gets filed into the corresponding person's folder. All these folders are automatically created for all recognizable faces and are housed within a "target_folders" directory.
- Show Recognizable Names Button: Displays a list of all the faces that the model can recognize.

## Requirements
Python 3.x
TensorFlow 2.x
PyTorch
facenet_pytorch
scikit-learn
matplotlib
OpenCV
albumentations
  
# GUI Overview

![GUI Start Page](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/4412c0f0-4575-45ba-9eef-bf64548d3862)

## Browse Image Button:
- Click to open the file explorer.
- Choose a face image for recognition. Ensure the face is listed among recognizable names (click "Show Recognizable Names" to view the list).
- After selecting "Open", the cropped facial region from the image is displayed.

Original Image:
![Original Image](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/0d02466e-e798-4ddc-bc29-0c9fd0fe1241)

Cropped Facial Region:

![Image Cropped to Face on GUI](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/ea7c5e02-a108-41be-a285-0ac9c0278996)

### Evaluate Button:
- Click to process the cropped image through the model.
- The model predicts the face's identity and displays it alongside a confidence score.
  
![GUI with prediction and confidence score](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/5dc193a2-f71a-4037-b601-60084f101fa5)

### Confirm Button:
- Click to confirm the model's prediction.
- The image is automatically saved into its corresponding person's folder. Each recognizable face has a pre-generated folder located under "target_folders", and a path to the saved image is provided for easy retrieval.

![GUI with path to image with file application showing image in correct folder and all faces' folders created](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/a1b82b82-3feb-40f7-8b01-9cf2a0af1d1f)

### Show Recognizable Names Button:
- Click to view a list of all faces that the model can recognize.
![image](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/b48d44dc-c7eb-4289-84d7-560aa9a4a270)
