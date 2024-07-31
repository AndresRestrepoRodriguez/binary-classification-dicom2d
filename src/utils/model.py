from src.models.DICOMBinaryClassification import BinaryClassificationCNN
import torch


def read_pytorch_model_eval(weights: str, device='cpu'):
    model = BinaryClassificationCNN()
    model.load_state_dict(torch.load(weights,  map_location=torch.device(device)))
    model.eval()

    return model