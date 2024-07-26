import torch
from models.DICOMBinaryClassification import BinaryClassificationCNN
import base64
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import io
from utils.data import (
    read_dicom_image,
    define_val_transformation
)
from utils.model import (
    read_pytorch_model_eval
)


def predict_model(image: str, model, image_base64=False, img_size=224, device='cpu'):

    if image_base64:
        dicom_data = base64.b64decode(image_base64)
        dicom_file = io.BytesIO(dicom_data)
    else:
        dicom_file = image
    dicom_image_array = read_dicom_image(dicom_file)
    transformation = define_val_transformation(img_size)
    dicom_image_transformed = transformation(dicom_image_array)
    dicom_image_transformed = dicom_image_transformed.to(device)

    model.eval()
    output = model(dicom_image_transformed).squeeze()
    predicted = torch.sigmoid(output).round()
    return predicted

