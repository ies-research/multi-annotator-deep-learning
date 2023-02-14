import numpy as np
import torch
import warnings

from pytorch_lightning import seed_everything

from skactiveml.utils import check_type

from sklearn.utils.validation import check_array, check_is_fitted

from torch.utils.data import DataLoader

from . import LightningClassifier
from ..modules import CoNALModule, MultiAnnotatorDataSet
from ..utils import check_annotator_features


class CoNALClassifier(LightningClassifier):
    """CoNALClassifier

    CoNAL (Common Noise Adaption Layers) [1] is an end-to-end learning solution with two types of noise adaptation
    layers: one is shared across annotators to capture their commonly shared confusions, and the other one is
    pertaining to each annotator to realize individual confusion.

    Parameters
    ----------
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined during the fit.
    missing_label : scalar or string or np.nan or None, optional (default=-1)
        Value to represent a missing label.
    cost_matrix : array-like, shape (n_classes, n_classes), optional (default=None)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class `classes[j]`  for a sample of class
        `classes[i]`. Can be only set, if `classes` is not `None`.
    random_state : int or None, optional (default=None)
        Determines random number for `fit` and `predict` method. Pass an int for reproducible results across multiple
        method calls. Defines also a global seed via `seed_everything`.

    References
    ----------
    [1] Chu, Z., Ma, J., & Wang, H. (2021, May). Learning from Crowds by Modeling Common Confusions.
        In AAAI (pp. 5832-5840).
    """

    def __init__(
        self,
        module_dict,
        trainer_dict=None,
        classes=None,
        missing_label=-1,
        cost_matrix=None,
        random_state=None,
    ):
        super(CoNALClassifier, self).__init__(
            module_class=CoNALModule,
            module_dict=module_dict,
            trainer_dict=trainer_dict,
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )

    def fit(
        self,
        X,
        y,
        A=None,
        X_val=None,
        y_val=None,
        data_loader_dict=None,
        val_data_loader_dict=None,
        transform=None,
        val_transform=None,
    ):
        """Fit the model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, *)
            The sample matrix `X` is the feature matrix. The exact form depends on the architecture of the underlying
            `module_class`.
        y : array-like of shape (n_samples, n_annotators)
            It contains the class labels of the training samples from the annotators.
        A : array-like of shape (n_samples, *), optional (default=None)
            The matrix `A` is the annotator feature matrix. The exact form depends on the `module_class`.
            If it is `None`, a one-hot encoding is used to differentiate between annotators.
        X_val : array-like of shape (n_samples, *) or None, optional (default=None)
            The sample matrix `X` is the feature matrix of the validation samples. The exact form depends on the
            modules on the `module_class`.
        y_val : array-like of shape (n_samples,) or None, optional (default=None)
            It contains the (true) class labels of the validation samples.
        data_loader_dict : dict or None, optional (default=None)
            Dictionary passed to the `DataLoader` object for the training data `X` and `y`. If `None`, an empty
            dictionary is passed.
        val_data_loader_dict : dict or None, optional (default=None)
            Dictionary passed to the `DataLoader` object for the validation data `X_val` and `y_val`. If `None`, an
            empty dictionary is passed.
        transform : Transform object or None, optional (default=None)
            Transforms to be applied to the training data `X`. If `None`, no transform is applied.
        val_transform : Transform object or None, optional (default=None)
            Transforms to be applied to the validation data `X_val`. If `None`, no transform is applied.

        Returns
        -------
        self: CoNALClassifier,
            `CoNALClassifier` object fitted on the training data.
        """
        # Check input parameters.
        X, y, _, X_val, y_val, data_loader_dict, val_data_loader_dict = self._validate_data(
            X=X,
            y=y,
            A=A,
            X_val=X_val,
            y_val=y_val,
            data_loader_dict=data_loader_dict,
            val_data_loader_dict=val_data_loader_dict,
        )

        # Set global seed for reproducibility.
        seed_everything(seed=self.random_state, workers=True)

        # Transform training data to tensors.
        train_data_set = MultiAnnotatorDataSet(X=X, y=y, transform=transform)
        train_data_loader = DataLoader(train_data_set, **data_loader_dict)

        # Setup validation data set and loader.
        val_data_loader = None
        if X_val is not None and y_val is not None:
            val_data_set = MultiAnnotatorDataSet(X=X_val, y=y_val, transform=val_transform)
            val_data_loader = DataLoader(val_data_set, **val_data_loader_dict)

        # Create CoNAL neural network.
        self.module_ = CoNALModule(**self.module_dict_)

        # Train CoNAL neural network.
        self.trainer_.fit(
            self.module_,
            train_dataloaders=train_data_loader,
            val_dataloaders=val_data_loader,
        )

        return self

    @torch.no_grad()
    def predict_proba(self, X, data_loader_dict=None, transform=None):
        """Returns class-membership probability estimates for the test data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, *)
            Test samples.
        data_loader_dict : dict or None (default=None)
            Dictionary passed to the DataLoader object for the training samples `X`. If `None`, an empty dictionary is
            passed.
        transform : Transform object or None, optional (default=None)
            Transforms to be applied to the test samples `X`. If `None`, no transform is applied.

        Returns
        -------
        P_class : numpy.ndarray of shape (n_samples, classes)
            `P_class[n, c]` is the probability, that instance `X[n]` belongs to the `classes_[c]`.
        """
        check_is_fitted(self)
        X = torch.tensor(check_array(X, **self._check_X_dict)).float()
        data_set = MultiAnnotatorDataSet(X=X, transform=transform)
        data_loader_dict = {} if data_loader_dict is None else data_loader_dict
        data_loader = DataLoader(data_set, **data_loader_dict)
        pred = self.trainer_.predict(self.module_, data_loader)
        P_class = torch.vstack([p[0] for p in pred]).numpy()
        return P_class

    @torch.no_grad()
    def predict_annotator_perf(
        self,
        X,
        return_confusion_matrix=False,
        data_loader_dict=None,
        transform=None,
    ):
        """Calculates the probability that an annotator provides the true label for a given sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, *)
            Test samples.
        return_confusion_matrix : bool, optional (default=False)
            If `return_confusion_matrix=True`, the entire confusion matrix per annotator is returned.
        data_loader_dict : dict or None (default=None)
            Dictionary passed to the DataLoader object for the training samples `X`. If `None`, an empty dictionary is
            passed.
        transform : Transform object or None, optional (default=None)
            Transforms to be applied to the test samples `X`. If `None`, no transform is applied.

        Returns
        -------
        P_perf : numpy.ndarray of shape (n_samples, n_annotators) or (n_samples, n_annotators, n_classes, n_classes)
            If `return_confusion_matrix=False`, `P_perf[n, m]` is the probability, that annotator `A[m]` provides the
            correct class label for sample `X[n]`. If `return_confusion_matrix=False`, `P_perf[n, m, c, j]` is the
            probability, that annotator `A[m]` provides the correct class label `classes_[j]` for sample `X[n]` and
            that this sample belongs to class `classes_[c]`. If `return_cond=True`, `P_perf[n, m, c, j]` is the
            probability that annotator `A[m]` provides the class label `classes_[j]` for sample `X[n]` conditioned that
            this sample belongs to class `classes_[c]`.
        """
        check_is_fitted(self)
        X = torch.tensor(check_array(X, **self._check_X_dict)).float()
        data_set = MultiAnnotatorDataSet(X=X, transform=transform)
        data_loader_dict = {} if data_loader_dict is None else data_loader_dict
        check_type(data_loader_dict, "data_loader_dict", dict)
        data_loader = DataLoader(data_set, **data_loader_dict)
        pred = self.trainer_.predict(self.module_, data_loader)
        P_class = torch.vstack([p[0] for p in pred]).numpy()
        P_annot = torch.vstack([p[1] for p in pred]).numpy()
        P_perf = np.array(
            [np.einsum("ij,ik->ijk", P_class, P_annot[:, :, i]) for i in range(self.n_annotators_)]
        ).swapaxes(0, 1)
        if return_confusion_matrix:
            return P_perf
        return P_perf.diagonal(axis1=-2, axis2=-1).sum(axis=-1)

    @torch.no_grad()
    def compute_annotator_embeddings(self, A=None):
        A = self.A_ if A is None else torch.tensor(A).float()
        return self.module_.ap_embed_a(A)

    @torch.no_grad()
    def _validate_data(self, X, y, A=None, X_val=None, y_val=None, data_loader_dict=None, val_data_loader_dict=None):
        X, y, X_val, y_val, data_loader_dict, val_data_loader_dict = super()._validate_data(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            data_loader_dict=data_loader_dict,
            val_data_loader_dict=val_data_loader_dict,
        )

        self.A_ = check_annotator_features(self.n_annotators_, A)
        self.A_ = torch.tensor(self.A_).float()

        if "n_classes" in self.module_dict_:
            warnings.warn("`n_classes` passed to `module_dict` will be overwritten by" " `len(self.classes_)`.")
        self.module_dict_["n_classes"] = len(self.classes_)

        if "n_annotators" in self.module_dict_:
            warnings.warn("`n_annotators` passed to `module_dict` will be overwritten by" " `self.n_annotators_`.")
        self.module_dict_["n_annotators"] = self.n_annotators_

        if "A" in self.module_dict_:
            warnings.warn("`A` passed to `module_dict` will be overwritten by" " `self.A_`.")
        self.module_dict_["A"] = self.A_

        return X, y, self.A_, X_val, y_val, data_loader_dict, val_data_loader_dict
