# Model Statistics, Information & Project Summary
![Accuracy and Loss of Model](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/b453a31c-6590-4f48-b676-032816770375)

# GUI

![GUI Start Page](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/4412c0f0-4575-45ba-9eef-bf64548d3862)

### Browse Image Button:
- Upon clicking, the files Application will appear - select file with a face to be recognized (make sure the face is recognizable by the model by clicking "Show Recognizable Names")

![GUI with File Application](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/21b5c698-c474-4182-86fb-1964d67aa377)
- Upon Selecting "Open" on files application, the cropped image of the face will appear

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

![GUI showing prediction and confidence score](https://github.com/kcoats4024/Facial-Detection-and-Recognition/assets/112397460/420fb032-3832-4d97-841b-9200ee7d2bc5)

### Show Recognizable Names Button:
- Upon clicking, a list of all all recognizable faces by the model will be displayed
