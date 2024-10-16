import torch
import base64
import numpy as np
import onnxruntime as ort
import io
from src.utils.data import (
    read_dicom_image,
    define_val_transformation,
    normalize_ct_int16
)
from typing import Union


def predict_model(image: str, model: torch.nn.Module, image_base64: bool = False, img_size: int = 224, device: str = 'cpu') -> float:
    """
    Predicts the output for a given DICOM image using a PyTorch model.

    Args:
        image (str): Path to the DICOM image or base64-encoded DICOM data.
        model (torch.nn.Module): The PyTorch model used for prediction.
        image_base64 (bool, optional): If True, the image is provided in base64 encoding. Defaults to False.
        img_size (int, optional): Size to which the input image is resized. Defaults to 224.
        device (str, optional): The device to run the model on (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        float: The model's prediction as a single floating-point value.
    """

    if image_base64:
        dicom_data = base64.b64decode(image)
        dicom_file = io.BytesIO(dicom_data)
    else:
        dicom_file = image
    dicom_image_array = read_dicom_image(dicom_file)
    transformation = define_val_transformation(img_size)
    dicom_image_transformed = transformation(dicom_image_array)
    dicom_image_transformed = dicom_image_transformed.to(device)

    #model.eval()
    with torch.no_grad():
        output = model(dicom_image_transformed.unsqueeze(0)).squeeze()
        #predicted = torch.sigmoid(output).round()
    return output.item()


def predict_model_citadel(image_data: Union[np.ndarray, list], model: torch.nn.Module, img_size: int = 224, device: str = 'cpu') -> list:
    """
    Predicts the output for a given image (CT scan) using a PyTorch model.

    Args:
        image_data (Union[np.ndarray, list]): The CT image data (either as a NumPy array or list).
        model (torch.nn.Module): The PyTorch model used for prediction.
        img_size (int, optional): Size to which the input image is resized. Defaults to 224.
        device (str, optional): The device to run the model on (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        list: The model's prediction as a list of floating-point values.
    """
    #img_array = np.asarray(image_data).astype(np.float32) / 255.0
    img_array = normalize_ct_int16(np.asarray(image_data))
    transformation = define_val_transformation(img_size)
    dicom_image_transformed = transformation(img_array)
    dicom_image_transformed = dicom_image_transformed.to(device)
    with torch.no_grad():
        output = model(dicom_image_transformed.unsqueeze(0)).squeeze()
    return output.item()
