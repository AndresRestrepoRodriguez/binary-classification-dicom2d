import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the function that exports the model to both formats
def export_model_pytorch(model, torchscript_file_path):
    """
    Exports a PyTorch model to both TorchScript and ONNX formats.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        input_shape (tuple): The shape of the dummy input tensor for ONNX export.
        torchscript_file_path (str): File path where the TorchScript model will be saved.
        onnx_file_path (str): File path where the ONNX model will be saved.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Export to TorchScript
    scripted_model = torch.jit.script(model)
    scripted_model.save(torchscript_file_path)
    print(f"TorchScript model saved to {torchscript_file_path}")