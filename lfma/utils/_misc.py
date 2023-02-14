import math
import numpy as np
import torch

from sklearn.utils import check_random_state


def concatenate_per_row(A, B):
    """
    Creates combinations of each row of matrix `A` with each row of matrix `B`.

    Parameters
    ----------
    A : array-like of shape (n_rows_a, n_cols_a)
        First matrix.
    B : array-like of shape (n_rows_b, n_cols_b)
        Second matrix.

    Returns
    -------
    out : numpy.ndarray of shape (n_rows_a * n_rows_b, n_cols_a + n_cols_b)
        Concatenated and combined matrices.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    m1, n1 = A.shape
    m2, n2 = B.shape
    out = np.zeros((m1, m2, n1 + n2), dtype=A.dtype)
    out[:, :, :n1] = A[:, None, :]
    out[:, :, n1:] = B
    return out.reshape(m1 * m2, -1)


def introduce_missing_annotations(y, percentage, missing_label=-1, random_state=None):
    """
    Randomly replace a specified proportion of annotations with `missing_label`.

    Parameters
    ----------
    y : numpy.ndarray of shape (n_samples, n_annotators)
        Annotation of each annotator for each sample.
    percentage : float in [0, 1]
        Percentage of annotations that should be replaced with `missing_label`.
    missing_label : scalar or string or np.nan or None, optional (default=-1)
        Value to represent a missing label.
    random_state : int or None, optional (default=None)
        Determines seed for reproducible results.


    Returns
    -------
    y_missing : numpy.ndarray of shape (n_samples, n_annotators)
        Annotations with randomly introduced missing labels.
    """
    y_missing = np.array(y.copy(), dtype=float)

    if percentage >= 1:
        no_missing_annotations = (y.shape[0] * y.shape[1]) - (y.shape[0] * percentage)
    else:
        no_missing_annotations = round((y.shape[0] * y.shape[1]) * percentage)

    is_missing = np.zeros(y.shape[0] * y.shape[1], dtype=bool)
    is_missing[:no_missing_annotations] = 1

    random_state = check_random_state(random_state)
    random_state.shuffle(is_missing)

    is_missing = is_missing.reshape(y.shape)

    y_missing[is_missing] = missing_label

    return y_missing


def cosine_similarity(A, B=None, eps=1e-8, gamma=1):
    """
    Compute cosine similarity (normalized to the interval [0, 1]) between each row of matrix `A` with each row of
    matrix `B`.

    Parameters
    ----------
    A : torch.Tensor of shape (n_rows_A, n_cols_A)
        First matrix.
    B : torch.Tensor of shape (n_rows_B, n_cols_B), optional (default=None)
        Second matrix. If `B=None`, the similarities are computed between each pair of rows in matrix `A`.
    eps : float > 0,
        Positive value to avoid division by zero.

    Returns
    -------
    S : torch.Tensor of shape (n_rows_a * n_rows_b, n_rows_a * n_rows_b)
        Computed cosine similarities.
    """
    B = A if B is None else B
    A_n, B_n = A.norm(dim=1)[:, None], B.norm(dim=1)[:, None]
    A_norm = A / torch.clamp(A_n, min=eps)
    B_norm = B / torch.clamp(B_n, min=eps)
    S = torch.mm(A_norm, B_norm.transpose(0, 1))
    S = torch.nan_to_num(torch.arccos(S)) / torch.pi
    return torch.exp(-gamma * S)


def rbf_kernel(A, B=None, gamma=None):
    """
    Compute radial basis function (RBF) kernel between each row of matrix `A` with each row of matrix `B`.

    Parameters
    ----------
    A : torch.Tensor of shape (n_rows_A, n_cols_A)
        First matrix.
    B : torch.Tensor of shape (n_rows_B, n_cols_B)
        Second matrix. If `B=None`, the similarities are computed between each pair of rows in matrix `A`.
    gamma : float >= 0,
        Bandwidth controlling the width of the RBF kernel. If `None`, we use the mean bandwidth criterion [1] to set a
        default value.

    Returns
    -------
    S : torch.Tensor of shape (n_rows_a * n_rows_b, n_rows_a * n_rows_b)
        Computed similarities via RBF kernel.

    References
    ----------
    [1] Chaudhuri, A., Kakde, D., Sadek, C., Gonzalez, L. and Kong, S., 2017, November. The mean and median criteria
        for kernel bandwidth selection for support vector data description. In 2017 IEEE International Conference on
        Data Mining Workshops (ICDMW) (pp. 842-849). IEEE.
    """

    if gamma is None:
        n_samples = len(A) + (0 if B is None else len(B))
        var_sum = A.var(0).sum() if B is None else torch.vstack((A, B)).var(0).sum()
        s_2 = (2 * n_samples * var_sum) / ((n_samples - 1) * math.log((n_samples - 1) / (2 * 1e-12)))
        gamma = 0.5 / s_2
    B = A if B is None else B
    return torch.exp(-gamma * torch.cdist(A, B) ** 2)
