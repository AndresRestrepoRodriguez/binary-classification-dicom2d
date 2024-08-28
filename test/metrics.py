import numpy as np
from scipy.special import rel_entr
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Taken from scikitplot
def aucrc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under Cumulative Gains Curve.

    Args:
        y_true (np.ndarray): ground truth
        y_score (np.ndarray): predicted probabilities/scores

    Returns:
        float: area under the RC curve
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    # make y_true a boolean vector
    y_true = y_true == 1
    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)
    value_range = np.arange(start=1, stop=len(y_true) + 1)
    gains = gains / float(np.sum(y_true))
    percentages = value_range / float(len(y_true))
    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])
    return auc(percentages, gains)


def auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area Under Precision Recall Curve.

    Args:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predicted probabilities/scores

    Returns:
        float: area under the PR curve
    """
    return average_precision_score(y_true, y_pred)


def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area Under Reciving Operator Curve.

    Args:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predicted probabilities/scores

    Returns:
        float: area under the RO curve
    """
    return roc_auc_score(y_true, y_pred)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the Specificity of the predictions

    Args:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predicted probabilities/scores

    Returns:
        float: specificity
    """
    yp = y_pred >= 0.5
    tn, fp, _, _ = confusion_matrix(y_true, yp).ravel()
    specificity = tn / (tn + fp)
    return specificity


def confusionmatrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Confusion Matrix for multiclass task.

    Args:
        y_true (np.ndarray): ground truth
        y_pred (np.ndarray): predicted probabilities/scores

    Returns:
        np.ndarray: confusion matrix as in sklearn
    """
    yp = y_pred >= 0.5
    return confusion_matrix(y_true, yp)


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary accuracy metric.

    \\[
    \\mbox{Acc} = \\frac{\\mbox{TP}}{\\mbox{P}+\\mbox{N}}
    \\]

    Examples:
        >>> binary_accuracy(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]))
        0.75
        >>> binary_accuracy(np.array([0, 0, 1, 1]), np.array([0.3, 0.8, 0.9, 0.7]))
        0.75

    Args:
        y_true (np.ndarray): numpy array containing the ground truth.
        y_pred (np.ndarray): numpy array containing the predictions.

    Returns:
        float: binary accuracy of the predictions.
    """

    yp = y_pred >= 0.5
    return accuracy_score(y_true, yp)


def binary_precision(y_true: np.ndarray, y_pred: np.ndarray):
    """Binary precision metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth.
        y_pred (np.ndarray): numpy array containing the predictions.

    Returns:
        float: binary precision of the predictions.
    """
    yp = y_pred >= 0.5
    # TODO: check what to do when y_pred has not predicted any class 1
    return precision_score(y_true, yp, zero_division=0.0)


def binary_recall(y_true: np.ndarray, y_pred: np.ndarray):
    """Binary recall metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth.
        y_pred (np.ndarray): numpy array containing the predictions.

    Returns:
        float: binary recall of the predictions.
    """
    yp = y_pred >= 0.5
    return recall_score(y_true, yp)


def binary_f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    """Binary f1-score metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth.
        y_pred (np.ndarray): numpy array containing the predictions.

    Returns:
        float: binary f1-score of the predictions.
    """
    yp = y_pred >= 0.5
    return f1_score(y_true, yp)


def binary_f05_score(y_true: np.ndarray, y_pred: np.ndarray):
    """Binary f_beta-score (beta = 0.5) metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth.
        y_pred (np.ndarray): numpy array containing the predictions.

    Returns:
        float: binary f0.5-score of the predictions.
    """
    yp = y_pred >= 0.5
    return fbeta_score(y_true, yp, beta=0.5)


def binary_f2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """Binary f_beta-score (beta = 2) metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth.
        y_pred (np.ndarray): numpy array containing the predictions.

    Returns:
        float: binary f2-score of the predictions.
    """
    yp = y_pred >= 0.5
    return fbeta_score(y_true, yp, beta=2)


def kl_divergence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary KL-Divergence

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        float: _description_
    """
    yp = y_pred >= 0.5
    p = np.array(np.unique(y_true, return_counts=True)[1], dtype=float)
    q = np.array(np.unique(yp, return_counts=True)[1], dtype=float)
    p /= np.array(np.sum(p), dtype=float)
    q /= np.array(np.sum(q), dtype=float)

    return np.sum(rel_entr(p, q))


###############################################################################
# 24027 metrics
###############################################################################


# True Positive Rate = Recall
def tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yp = y_pred >= 0.5
    return recall_score(y_true, yp)


# False Positive Rate
def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusionmatrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


# False Negative Rate
def fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, fn, tp = confusionmatrix(y_true, y_pred).ravel()
    return fn / (fn + tp)


# True Negative Rate
def tnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusionmatrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


# False Discovery Rate
def fdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, _, tp = confusionmatrix(y_true, y_pred).ravel()
    return fp / (fp + tp)


# Negative Predictive Value
def npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusionmatrix(y_true, y_pred).ravel()
    return tn / (tn + fn)


# False Omission Rate
def false_omission_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusionmatrix(y_true, y_pred).ravel()
    return 1 - tn / (tn + fn)


# Positive Likelihood Ratio
def positive_likelihood_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return tpr(y_true, y_pred) / fpr(y_true, y_pred)


# Negative Likelihood Ratio
def negative_likelihood_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return fnr(y_true, y_pred) / tnr(y_true, y_pred)


def demographic_parity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yp = y_pred > 0.5
    yp_cl = yp == 1
    positive_predictions = np.sum(yp_cl)
    total_instances = len(y_true)
    ppr = positive_predictions / total_instances
    # npr = 1 - ppr
    return ppr  # min(ppr, npr)


def matthews_correlation(y_true, y_pred):
    yp = y_pred >= 0.5
    return matthews_corrcoef(y_true, yp)


def balanced_accuracy(y_true, y_pred):
    yp = y_pred >= 0.5
    return balanced_accuracy_score(y_true, yp)


def diagnostic_odd_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Diagnostic Odd Ratio is defined as

    \\[
        \\mbox{DOR} = \\frac{\\mbox{LR}+}{\\mbox{LR}-}
    \\]

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        List[float]: a list containing Diagnostic Odd Ratio per class
    """
    return positive_likelihood_ratio(y_true, y_pred) / negative_likelihood_ratio(
        y_true, y_pred
    )


def prevalence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per class Prevalence metric.

    Args:
        y_true (np.ndarray): numpy array containing the ground truth
        y_pred (np.ndarray): numpy array containing the predictions

    Returns:
        List[float]: a list containing Prevalence per class
    """
    total = len(y_pred)
    return np.sum(y_true == 1) / total