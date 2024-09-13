from torch.utils.data import Dataset
import os
import pydicom
import numpy as np
from utils.data import normalize_image, normalize_ct_int16
from typing import List, Callable, Optional


class BinaryDICOMDataset(Dataset):
    """
    A PyTorch Dataset class for loading and managing a binary classification dataset from DICOM images.

    This dataset assumes that the images are organized in subdirectories, with each subdirectory representing a class.
    The directory structure should look like:
    - directory/
        - class_1/
            - image_1.dcm
            - image_2.dcm
            - ...
        - class_2/
            - image_1.dcm
            - image_2.dcm
            - ...

    Attributes:
        directory (str): The root directory containing the dataset.
        transform (Optional[Callable]): An optional transform to apply to the images.
        filenames (List[str]): List of full file paths to the DICOM images.
        labels (List[int]): List of labels corresponding to the image classes.

    Args:
        directory (str): The root directory containing the dataset organized into subdirectories by class.
        classes (List[str]): List of class names that correspond to the subdirectory names.
        transform (Optional[Callable], optional): Optional transformation to be applied to the images. Defaults to None.
    """

    def __init__(self, directory: str, classes: List[str], transform: Optional[Callable] = None):
        self.directory = directory
        self.transform = transform
        self.filenames: List[str] = []
        self.labels: List[int] = []

        # Load all images and labels from the directory structure
        for label, subdir in enumerate(classes):
            subdir_path = os.path.join(directory, subdir)
            for filename in os.listdir(subdir_path):
                if filename.endswith('.dcm') or filename.endswith('.DCM'):
                    self.filenames.append(os.path.join(subdir_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        ds = pydicom.dcmread(self.filenames[idx])
        image = ds.pixel_array
        image = normalize_ct_int16(image)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    



