import torch
from torch.utils.data import Dataset


class MultiAnnotatorDataSet(Dataset):
    """MultiAnnotatorDataSet

    Dataset to deal with samples annotated by multiple annotators.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, *)
        Samples' features whose shape depends on the concrete learning problem.
    y : torch.Tensor of shape (n_samples, *)
        Samples' targets whose shape depends on the concrete learning problem.
    A: torch.Tensor of shape (n_annotators, *)
        Annotators' features whose shape depends on the concrete learning problem.
    """

    def __init__(self, X, y=None, A=None, transform=None):
        super(MultiAnnotatorDataSet).__init__()
        self.X = X
        self.y = y
        self.A = A
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.transform(self.X[idx]) if self.transform else self.X[idx]
        if self.y is not None and self.A is not None:
            return x, self.y[idx], self.A
        elif self.y is None and self.A is not None:
            return x, self.A
        elif self.y is not None and self.A is None:
            return x, self.y[idx]
        else:
            return x


class EMMultiAnnotatorDataSet(MultiAnnotatorDataSet):
    """MultiAnnotatorDataSet

    Dataset to deal with samples annotated by multiple annotators. It is particularly designed for models trained via
    the EM-algorithm, where targets are iteratively updated.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, *)
        Samples' features whose shape depends on the concrete learning problem.
    y : torch.Tensor of shape (n_samples, *)
        Samples' targets whose shape depends on the concrete learning problem.
    y_est : torch.Tensor of shape (n_samples, *)
        Estimated samples' targets whose shape depends on the concrete learning problem.
    A: torch.Tensor of shape (n_annotators, *)
        Annotators' features whose shape depends on the concrete learning problem.
    """

    def __init__(self, X, y=None, y_est=None, A=None, transform=None):
        super(EMMultiAnnotatorDataSet).__init__(X=X, y=y, A=A, transform=transform)
        self.y_est = torch.zeros_like(y).float() if y_est is None else y_est

    def __getitem__(self, idx):
        x = self.transform(self.X[idx]) if self.transform else self.X[idx]
        if self.y is not None and self.A is not None:
            return x, self.y[idx], self.y_est[idx], self.A
        elif self.y is None and self.A is not None:
            return x, self.A
        elif self.y is not None and self.A is None:
            return x, self.y[idx], self.y_est[idx]
        else:
            return x
