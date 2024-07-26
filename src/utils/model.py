from models.DICOMBinaryClassification import BinaryClassificationCNN
import torch


def read_pytorch_model_eval(weights: str, device='cpu'):
    model = BinaryClassificationCNN()
    model.load_state_dict(torch.load(weights))
    model.to(device)
    model.eval()

    return model