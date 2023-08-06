# Model Statistics & Project Summary
![Accuracy and Loss of Model](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/b453a31c-6590-4f48-b676-032816770375)

### Graphical User Interface (GUI) Summary:

- Browse Image Button: Allows users to select an image file to be recognized. The model can identify if the face in the image is one of the recognizable faces.
- Evaluate Button: Once an image is selected, this button processes the image through the model. The prediction, along with a confidence score, is then displayed.
- Confirm Button: Once a prediction is made, users can confirm its accuracy. The image then gets filed into the corresponding person's folder. All these folders are automatically created for all recognizable faces and are housed within a "target_folders" directory.
- Show Recognizable Names Button: Displays a list of all the faces that the model can recognize.
  
# GUI Overview

![GUI Start Page](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/4412c0f0-4575-45ba-9eef-bf64548d3862)

### Browse Image Button:
- Click to open the file explorer.
- Choose a face image for recognition. Ensure the face is listed among recognizable names (click "Show Recognizable Names" to view the list).
- After selecting "Open", the cropped facial region from the image is displayed.

Original:
![Original Image](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/0d02466e-e798-4ddc-bc29-0c9fd0fe1241)

Cropped to face:

![Image Cropped to Face on GUI](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/ea7c5e02-a108-41be-a285-0ac9c0278996)

- Now the Face is ready to be evaluated via the "Evaluate" button

### Evaluate Button:
- Upon clicking, the image is ran through the model and a prediction of the name of the face along with the confidence score is displayed:
![GUI with prediction and confidence score](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/5dc193a2-f71a-4037-b601-60084f101fa5)
- Now the evaluation is ready to be confirmed via the "Confirm" button

### Confirm Button:
- Upon clicking (confirming the prediction is correct), the image will be filed into the corresponding person's folder, which has already been automatically created for all recognizable faces. All of these folders are in folder "target_folders"

![GUI with path to image with file application showing image in correct folder and all faces' folders created](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/a1b82b82-3feb-40f7-8b01-9cf2a0af1d1f)

### Show Recognizable Names Button:
- Upon clicking, a list of all all recognizable faces by the model will be displayed
![image](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/b48d44dc-c7eb-4289-84d7-560aa9a4a270)
