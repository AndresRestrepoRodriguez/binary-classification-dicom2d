import requests
import base64
import glob as glob
import os
import pydicom
import numpy as np
from PIL import Image
import torch
import pandas as pd
import metrics as metrics_citadel


images_brain = glob.glob(os.path.join('data/val/brain', '*.dcm'))
images_chest = glob.glob(os.path.join('data/val/chest', '*.dcm'))
tmp_image_path = 'data/tmp_png_dicom.png'


y_true = []
y_pred = []
y_score = []

def dicom_to_png(dicom_file_path, png_file_path):
    # Read the DICOM file
    dicom_image = pydicom.dcmread(dicom_file_path, force=True)
    
    # Get the pixel array from the DICOM file
    image_array = dicom_image.pixel_array
    
    # Normalize the pixel array to the range 0-255
    #image_array_normalized = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
    #image_array_normalized = image_array_normalized.astype(np.uint8)
    
    # Create an Image object from the pixel array
    image = Image.fromarray(image_array)
    
    # Save the image as a PNG file
    image.save(png_file_path)
    #print(f"Converted {dicom_file_path} to {png_file_path}")


for image in images_brain:
    ground_true_class = 0
    label = 'brain'
    dicom_to_png(image, tmp_image_path)

    with open(tmp_image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    data = {'image': encoded_image}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    prob = response.json().get('predictions')
    prob = np.float32(prob)
    print(prob)
    print(type(prob))
    #class_prob = torch.sigmoid(torch.tensor(1-prob)).item()
    y_pred.append(prob)
    y_true.append(ground_true_class)
    y_score.append(prob)


for image in images_chest:
    ground_true_class = 1
    label = 'chest'
    dicom_to_png(image, tmp_image_path)

    with open(tmp_image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    data = {'image': encoded_image}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    prob = response.json().get('predictions')
    prob = np.float32(prob)
    #class_prob = torch.sigmoid(torch.tensor(prob)).item()
    print(prob)
    print(type(prob))
    y_pred.append(prob)
    y_true.append(ground_true_class)
    y_score.append(prob)


y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)
y_score_np = np.array(y_score)

binary_accuracy = metrics_citadel.binary_accuracy(y_true_np, y_pred_np)
aucrc = metrics_citadel.aucrc(y_true_np, y_score_np)
auprc = metrics_citadel.auprc(y_true, y_score)
auroc = metrics_citadel.auroc(y_true_np, y_score_np)
confusionmatrix = metrics_citadel.confusionmatrix(y_true_np, y_pred_np)
binary_precision = metrics_citadel.binary_precision(y_true_np, y_pred_np)
binary_recall = metrics_citadel.binary_recall(y_true_np, y_pred_np)
binary_f1_score = metrics_citadel.binary_f1_score(y_true_np, y_pred_np)
binary_f05_score = metrics_citadel.binary_f05_score(y_true_np, y_pred_np)
binary_f2_score = metrics_citadel.binary_f2_score(y_true_np, y_pred_np)
false_omission_rate = metrics_citadel.false_omission_rate(y_true_np, y_pred_np)
positive_likelihood_ratio = metrics_citadel.positive_likelihood_ratio(y_true_np, y_pred_np)
negative_likelihood_ratio = metrics_citadel.negative_likelihood_ratio(y_true_np, y_pred_np)
prevalence = metrics_citadel.prevalence(y_true_np, y_pred_np)

print(f"Binary Accuracy: {binary_accuracy:.4f}")
print(f"AUCRC: {aucrc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"AUROC: {auroc:.4f}")
print(f"Confusion Matrix: {confusionmatrix}")
print(f"Binary Precision: {binary_precision:.4f}")
print(f"Binary Recall: {binary_recall:.4f}")
print(f"Binary F1 Score: {binary_f1_score:.4f}")
print(f"Binary F0.5 Score: {binary_f05_score:.4f}")
print(f"Binary F2 Score: {binary_f2_score:.4f}")
print(f"False Omission Rate: {false_omission_rate:.4f}")
print(f"Positive Likelihood Ratio: {positive_likelihood_ratio:.4f}")
print(f"Negative Likelihood Ratio: {negative_likelihood_ratio:.4f}")
print(f"Prevalence: {prevalence:.4f}")
