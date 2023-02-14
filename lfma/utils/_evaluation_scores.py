import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.utils.validation import check_array


def micro_accuracy_score(y_true, y_pred):
    """
    Compute micro average classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or of shape (n_samples, n_annotators)
        True class labels.
    y_pred : array-like of shape (n_samples,) or of shape (n_samples, n_annotators)
        Predicted class labels.

    Returns
    -------
    acc : float in [0, 1]
        Micro average classification accuracy score.
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    return accuracy_score(y_true=y_true.ravel(), y_pred=y_pred.ravel())


def macro_accuracy_score(y_true, y_pred):
    """
    Compute macro average classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or of shape (n_samples, n_annotators)
        True class labels.
    y_pred : array-like of shape (n_samples,) or of shape (n_samples, n_annotators)
        Predicted class labels.

    Returns
    -------
    acc : float in [0, 1]
        Macro average classification accuracy score.
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if y_true.ndim == 1:
        return balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    else:
        n_annotators = y_true.shape[1]
        macro_acc = 0
        for a_idx in range(n_annotators):
            macro_acc += balanced_accuracy_score(y_true=y_true[:, a_idx], y_pred=y_pred[:, a_idx])
        return macro_acc / n_annotators


def cross_entropy_score(y_true, y_pred):
    """
    Compute cross-entropy between true class labels and predicted class-membership probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or of shape (n_samples, n_annotators)
        True class labels.
    y_pred : array-like of shape (n_samples, n_classes) or of shape (n_samples, n_annotators, n_classes)
        Predicted class-membership probabilities.

    Returns
    -------
    cross_entropy : float
        Cross-entropy score.
    """
    y_true = check_array(y_true, ensure_2d=False, allow_nd=True)
    y_pred = check_array(y_pred, ensure_2d=False, allow_nd=True)
    if y_true.ndim == 1:
        return log_loss(y_true=y_true, y_pred=y_pred)
    else:
        return log_loss(y_true=y_true.ravel(), y_pred=y_pred.reshape(-1, y_pred.shape[-1]))


def brier_score(y_true, y_pred):
    """
    Compute Brier score between true class labels and predicted class-membership probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or of shape (n_samples, n_annotators)
        True class labels.
    y_pred : array-like of shape (n_samples,) or of shape (n_samples, n_annotators, n_classes)
        Predicted class-membership probabilities.

    Returns
    -------
    brier : float
        Brier score.
    """
    y_true = check_array(y_true, ensure_2d=False, allow_nd=True)
    y_pred = check_array(y_pred, ensure_2d=False, allow_nd=True)
    if y_true.ndim == 2:
        y_true = y_true.ravel()
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    n_classes = y_pred.shape[-1]
    y_true = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
