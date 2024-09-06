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
import imageio


images_bodyfront = glob.glob(os.path.join('data/val/bodyfront', '*.dcm'))
images_bodyside = glob.glob(os.path.join('data/val/bodyside', '*.dcm'))
tmp_image_path = 'data/tmp_png_dicom.png'


y_true = []
y_pred = []
y_score = []

def read_dicom_as_int16(file_path):
    """
    Reads a DICOM image and converts it to an int16 NumPy array.
    
    Parameters:
    - file_path: Path to the DICOM file.
    
    Returns:
    - image_array: Image as a NumPy array with int16 data type.
    """
    dicom_ds = pydicom.dcmread(file_path)
    image_array = dicom_ds.pixel_array.astype(np.int16)
    return image_array


def save_int16_as_png_fixed(image_array, output_path):
    """
    Saves an int16 NumPy array as a PNG image using a fixed normalization based on the int16 range.
    
    Parameters:
    - image_array: The NumPy array to save as a PNG.
    - output_path: Path to save the PNG image.
    """
    # Define the fixed int16 range
    min_val = np.iinfo(np.int16).min  # -32768
    max_val = np.iinfo(np.int16).max  # 32767
    
    # Normalize the image to the range [0, 255]
    image_array_normalized = (image_array - min_val) / (max_val - min_val)
    image_array_normalized = (image_array_normalized * 255).astype(np.uint8)
    
    imageio.imwrite(output_path, image_array_normalized)
    return image_array_normalized


def png_to_base64(png_path):
    """
    Converts a PNG image to a base64-encoded string.
    
    Parameters:
    - png_path: Path to the PNG image.
    
    Returns:
    - base64_string: Base64-encoded string of the PNG image.
    """
    with open(png_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string


for image in images_bodyfront:
    ground_true_class = 0
    label = 'brain'

    image_array = read_dicom_as_int16(image)
    save_int16_as_png_fixed(image_array, tmp_image_path)
    base64_png_string = png_to_base64(tmp_image_path)

    data = {'image': base64_png_string}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    prob = response.json().get('predictions')


    prob = np.float32(prob)
    print(prob)
    print(type(prob))
    #class_prob = torch.sigmoid(torch.tensor(1-prob)).item()
    y_pred.append(prob)
    y_true.append(ground_true_class)
    y_score.append(prob)


for image in images_bodyside:
    ground_true_class = 1
    label = 'brain'

    image_array = read_dicom_as_int16(image)
    save_int16_as_png(image_array, tmp_image_path)
    base64_png_string = png_to_base64(tmp_image_path)

    data = {'image': base64_png_string}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    prob = response.json().get('predictions')

    prob = np.float32(prob)
    print(prob)
    print(type(prob))
    #class_prob = torch.sigmoid(torch.tensor(1-prob)).item()
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
