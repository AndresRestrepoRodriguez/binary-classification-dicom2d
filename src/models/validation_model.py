import torch
from models.DICOMBinaryClassification import BinaryClassificationCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def validate_model(data_loader, model_path, model_type='pytorch'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if model_type == 'pytorch':
            model = BinaryClassificationCNN()
            model.load_state_dict(torch.load(model_path))
            model.to(device)

        elif model_type == 'torchscript':
            model = torch.jit.load(model_path)
            model.to(device)
    except Exception as e:
        raise Exception(f'Problem loading the Model: {e}')

    """elif model_type == 'onnx':
        session = onnxruntime.InferenceSession(model_path)
        def model(x):
            inputs = {session.get_inputs()[0].name: x.numpy()}
            outputs = session.run(None, inputs)
            return torch.from_numpy(outputs[0])"""

    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)

            if model_type == 'onnx':
                outputs = model(images.cpu())
            else:
                outputs = model(images).squeeze()

            predicted = torch.sigmoid(outputs).round()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)

    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)