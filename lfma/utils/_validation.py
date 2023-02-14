import numpy as np


def check_annotator_features(n_annotators, A=None):
    """
    Check's the dimension and converts the data type of the annotator features.

    Parameters
    ----------
    n_annotators : int
        Number of annotators.
    A : array-like of shape (n_annotators, n_annot_features), optional (default=None)
        If None, for each annotator a one-hot encoded feature is created. Otherwise, the data type is converted and
        the dimension is checked.

    Returns
    -------
    A : numpy.ndarray of shape (n_annotators, n_annot_features)
        Check annotator features.
    """
    if A is None:
        # One-hot encoding of annotators.
        A = np.eye(n_annotators).astype(np.float32)
    else:
        A = np.asarray(A).astype(np.float32)
        if A.shape[0] != n_annotators:
            raise ValueError(
                f"The number of rows in `A` must be equal "
                f"to the number of columns in `y`. "
                f"Got `A.shape[0]={A.shape[0]}, "
                f"y.shape[1]={n_annotators}` instead."
            )
    return A
