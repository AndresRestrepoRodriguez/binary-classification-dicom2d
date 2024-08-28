import torch
from models.DICOMBinaryClassification import BinaryClassificationCNN
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_recall_curve,
    auc,
    fbeta_score,


)
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from utils import metrics as metrics


def validate_model(data_loader, model_path, classes, model_type='pytorch'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if model_type == 'pytorch':
            model = BinaryClassificationCNN()
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

        elif model_type == 'torchscript':
            model = torch.jit.load(model_path)
            model.to(device)
            model.eval()

        elif model_type == 'onnx':
            ort_session = ort.InferenceSession(model_path)

            def model(x):
                # Preparing input to the ONNX model
                inputs = {ort_session.get_inputs()[0].name: x.numpy()}
                # Running inference
                outputs = ort_session.run(None, inputs)
                # Converting output from ONNX to tensor for consistency with PyTorch models
                return torch.from_numpy(outputs[0])
        else:
            raise ValueError("Invalid model type specified.")
    except Exception as e:
        raise Exception(f'Problem loading the Model: {e}')

    
    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for images, labels in data_loader:

            # Adjust device handling for ONNX as it needs CPU numpy arrays
            if model_type == 'onnx':
                images = images.to('cpu')
                outputs = model(images)
            else:
                images = images.to(device)
                outputs = model(images).squeeze()
            
            all_scores.extend(outputs.cpu().numpy())
            predicted = torch.sigmoid(outputs).round()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_score = np.array(all_scores)

    binary_accuracy = metrics.binary_accuracy(y_true, y_pred)
    aucrc = metrics.aucrc(y_true, y_score)
    auprc = metrics.auprc(y_true, y_score)
    auroc = metrics.auroc(y_true, y_score)
    confusionmatrix = metrics.confusionmatrix(y_true, y_pred)
    binary_precision = metrics.binary_precision(y_true, y_pred)
    binary_recall = metrics.binary_recall(y_true, y_pred)
    binary_f1_score = metrics.binary_f1_score(y_true, y_pred)
    binary_f05_score = metrics.binary_f05_score(y_true, y_pred)
    binary_f2_score = metrics.binary_f2_score(y_true, y_pred)
    false_omission_rate = metrics.false_omission_rate(y_true, y_pred)
    positive_likelihood_ratio = metrics.positive_likelihood_ratio(y_true, y_pred)
    negative_likelihood_ratio = metrics.negative_likelihood_ratio(y_true, y_pred)
    prevalence = metrics.prevalence(y_true, y_pred)

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
    

    