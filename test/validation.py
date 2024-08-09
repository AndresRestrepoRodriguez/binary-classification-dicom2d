import pandas as pd
import numpy as np
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
    fbeta_score
)


def get_predictions(dataframe):

    all_labels = dataframe['true_class'].values
    all_predictions = dataframe['pred_class'].values
    return all_labels, all_predictions


def get_metrics(all_labels, all_predictions, classes):
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


if __name__ == "__main__":
    classes = ['brain', 'chest']
    csv_path = 'results_torchcript_no_norm.csv'
    predictions_df = pd.read_csv(csv_path)
    true_labels, pred_labels = get_predictions(predictions_df)
    get_metrics(true_labels, pred_labels, classes)

