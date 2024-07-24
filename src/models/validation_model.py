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

    with torch.no_grad():
        for images, labels in data_loader:

            # Adjust device handling for ONNX as it needs CPU numpy arrays
            if model_type == 'onnx':
                images = images.to('cpu')
                outputs = model(images)
            else:
                images = images.to(device)
                outputs = model(images).squeeze()
            
            predicted = torch.sigmoid(outputs).round()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    # AUROC
    auroc = roc_auc_score(all_labels, all_predictions)

    # AUPRC
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_predictions)
    auprc = auc(recall_vals, precision_vals)

    # F-beta scores
    f_beta_0_5 = fbeta_score(all_labels, all_predictions, beta=0.5)
    f_beta_2 = fbeta_score(all_labels, all_predictions, beta=2)

    # False Discovery Rate (FDR)
    fdr = fp / (fp + tp)

    # False Negative Rate (FNR)
    fnr = fn / (fn + tp)

    # False Omission Rate (FOR)
    for_ = fn / (fn + tn)

    # False Positive Rate (FPR)
    fpr = fp / (fp + tn)

    # Negative Predictive Value (NPV)
    npv = tn / (tn + fn)

    # Negative Likelihood Ratio (NLR)
    nlr = fnr / (tn / (tn + fp))

    # Positive Likelihood Ratio (PLR)
    plr = recall / fpr

    # Prevalence
    prevalence = (tp + fn) / (tp + tn + fp + fn)

    # True Negative Rate (TNR)
    tnr = tn / (tn + fp)


    accuracy = accuracy_score(all_labels,all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot().figure_.savefig('confusion_matrix.png')
    print(f"Confusion Matrix: {cm}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"auroc score: {auroc:.4f}")
    

    print(f"auprc: {auprc:.4f}")
    print(f"F-beta scores _0_5: {f_beta_0_5:.4f}")
    print(f"F-beta scores _2: {f_beta_2:.4f}")
    print(f"False Discovery Rate (FDR): {fdr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"False Omission Rate (FOR): {for_:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Negative Predictive Value (NPV): {npv:.4f}")
    print(f"Negative Likelihood Ratio (NLR): {nlr:.4f}")
    print(f"Positive Likelihood Ratio (PLR): {plr:.4f}")
    print(f"prevalence: {prevalence:.4f}")
    print(f"True Negative Rate (TNR): {tnr:.4f}")

    

    