import numpy as np
import torch

from abc import ABC

from copy import deepcopy

from pytorch_lightning import LightningModule, Trainer

from skactiveml.base import AnnotatorModelMixin, SkactivemlClassifier
from skactiveml.utils import check_type, is_labeled, is_unlabeled

from sklearn.utils.validation import (
    check_array,
    column_or_1d,
)


class LightningClassifier(SkactivemlClassifier, AnnotatorModelMixin, ABC):
    """LightningClassifier

    This class is an abstract implementation for classifiers using PyTorch Lightning modules as basis.

    Parameters
    ----------
    module_class : LightningModule
        Class implementation of a lightning module whose object is to be trained.
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined during the fit.
    missing_label : scalar or string or np.nan or None, optional (default=-1)
        Value to represent a missing label.
    cost_matrix : array-like, shape (n_classes, n_classes), optional (default=None)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class `classes[j]`  for a sample of class
        `classes[i]`. Can be only set, if `classes` is not `None`.
    random_state : int or None, optional (default=None)
        Determines random number for `predict` method. Pass an int for reproducible results across multiple method
        calls. Defines also a global seed via `seed_everything`.
    """

    def __init__(
        self,
        module_class,
        module_dict=None,
        trainer_dict=None,
        classes=None,
        missing_label=-1,
        cost_matrix=None,
        random_state=None,
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.module_class = module_class
        self.module_dict = module_dict
        self.trainer_dict = trainer_dict

    @torch.no_grad()
    def _validate_data(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        data_loader_dict=None,
        val_data_loader_dict=None,
        remove_unlabeled_samples=True,
    ):
        # Check `__init__` parameters.
        if not issubclass(self.module_class, LightningModule):
            raise TypeError("`module_class` must be a subclass of `LightningModule`")
        if self.trainer_dict is not None:
            check_type(self.trainer_dict, "trainer_dict", dict)
            self.trainer_dict_ = deepcopy(self.trainer_dict)
        else:
            self.trainer_dict_ = {}

        if self.module_dict is not None:
            check_type(self.module_dict, "module_dict", dict)
            self.module_dict_ = deepcopy(self.module_dict)
        else:
            self.module_dict_ = {}

        # Check `fit` parameters.
        self._check_X_dict = {
            "ensure_2d": False,
            "allow_nd": True,
            "dtype": np.float32,
        }
        X, y, _ = super(LightningClassifier, self)._validate_data(
            X=X,
            y=y,
            check_X_dict=self._check_X_dict,
            check_y_dict={"ensure_2d": True},
            y_ensure_1d=False,
        )
        if remove_unlabeled_samples:
            n_labels_per_sample = np.sum(is_labeled(y, missing_label=-1), axis=1)
            is_lbld = n_labels_per_sample > 0
            X = X[is_lbld]
            y = y[is_lbld]
        X = torch.tensor(X).float()
        y = torch.tensor(y).long()

        if X_val is not None:
            X_val = check_array(X_val, **self._check_X_dict)
            y_val = self._le.transform(y_val)
            y_val = column_or_1d(y_val)
            if np.any(is_unlabeled(y_val, missing_label=-1)):
                raise ValueError("`y_val` cannot contain missing labels.")
            X_val = torch.tensor(X_val).float()
            y_val = torch.tensor(y_val).long()

        # Check data loader dictionaries.
        data_loader_dict = {} if data_loader_dict is None else data_loader_dict
        val_data_loader_dict = {} if val_data_loader_dict is None else val_data_loader_dict
        check_type(data_loader_dict, "data_loader_dict", dict)
        check_type(val_data_loader_dict, "val_data_loader_dict", dict)

        # Specify number of annotators.
        self.n_annotators_ = y.shape[1]

        # Create trainer object.
        self.trainer_ = Trainer(**self.trainer_dict_)

        return X, y, X_val, y_val, data_loader_dict, val_data_loader_dict
