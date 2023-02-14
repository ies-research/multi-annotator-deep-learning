import numpy as np
import pandas as pd

from annotlib.utils import check_labelling_array

from itertools import combinations

from copy import deepcopy

from skactiveml.utils import is_labeled

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils import (
    check_X_y,
    check_scalar,
    check_random_state,
    check_array,
    column_or_1d,
)


def annot_sim_clf_model(
    X,
    y_true,
    n_annotators=5,
    classifiers=None,
    train_ratios="equidistant",
    random_state=None,
):
    """
    Annotators can be seen as human classifiers. Hence, we use classifiers based on machine learning techniques to
    represent these annotators. Given a data set comprising samples with their true labels, a classifier is trained on
    a subset of sample-label-pairs. Subsequently, this trained classifier is used as proxy of an annotator. As a
    result, the labels for a sample are provided by this classifier.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Samples of the whole data set.
    y_true: array-like, shape (n_samples)
        True class labels of the given samples X.
    n_annotators: int
        Number of annotators who are simulated.
    classifiers: object | list of objects of shape (n_annotators)
        This parameter is either a single classifier implementing the `fit` and `predict_proba` or a
        list of such classifiers. If the parameter is not a list, each annotator is simulated on the same
        classification model, whereas if it is a list, the annotators may be simulated on different classifier
        types or even different parametrizations. The default classifiers parameter is a list of Gaussian processes
        with the same parameters.
    train_ratios: 'one-hot' or 'equidistant' or array-like of shape (n_annotators, n_classes)
        The entry `train_ratios_[j, i]` indicates the ratio of samples of class i used for training the classifier of
        annotator j , e.g., `train_ratios_[2,4]=0.3`: 30% of the samples for class 4 are used to train the classifier
        of annotator with the id 2.
    random_state: None or int or numpy.random.RandomState, optional (default=None)
        The random state used for generating regression values of the annotators.

    Returns
    -------
    Y : np.ndarray of shape (n_samples, n_annotators)
        Class labels of simulated annotators.
    """
    # Check `X` and `y_true`.
    X, y_true = check_X_y(X, y_true, ensure_2d=False, allow_nd=True)
    n_samples = np.size(X, 0)

    # Check number of annotators.
    check_scalar(n_annotators, name="n_annotators", target_type=int)

    # Check and transform `random_state`.
    random_state = check_random_state(random_state)

    # Transform class labels to interval [0, n_classes-1].
    le = LabelEncoder().fit(y_true)
    y_transformed = le.transform(y_true)
    y_unique = np.unique(y_transformed)
    n_classes = len(y_unique)

    if not isinstance(classifiers, list):
        clf = GaussianProcessClassifier(random_state=random_state) if classifiers is None else classifiers
        classifiers = [deepcopy(clf) for _ in range(n_annotators)]
    for clf_idx, clf in enumerate(classifiers):
        if (
            len(classifiers) != n_annotators
            or getattr(clf, "predict_proba", None) is None
            or getattr(clf, "fit", None) is None
        ):
            raise TypeError(
                "The parameter `classifiers` must be single classifier or a "
                "list of classifiers supporting the methods :py:method::`fit` "
                "and :py:method::`predict_proba`."
            )
        else:
            classifiers[clf_idx] = deepcopy(clf)
    if isinstance(train_ratios, str):
        if train_ratios == "one-hot":
            train_ratios = np.empty((n_annotators, n_classes))
            class_indices = np.arange(0, n_classes)
            for a_idx in range(n_annotators):
                class_j = a_idx % n_classes
                train_ratios[a_idx, class_j] = 1
                train_ratios[a_idx, class_indices != class_j] = 0.2
        elif train_ratios == "equidistant":
            train_ratios = np.linspace(1 / n_annotators, 1, n_annotators)
            train_ratios = np.repeat(train_ratios, n_classes)
            train_ratios = train_ratios.reshape((n_annotators, n_classes))
    train_ratios_ = check_labelling_array(train_ratios, (n_annotators, n_classes), "train_ratios")

    # Simulate annotators.
    y = np.empty((n_samples, n_annotators))
    class_indices = [np.where(y_transformed == c)[0] for c in y_unique]
    for a_idx in range(n_annotators):
        train_size = [int(train_ratios_[a_idx, c] * len(class_indices[c]) + 0.5) for c in y_unique]
        train = [random_state.choice(class_indices[c], size=train_size[c], replace=False) for c in y_unique]
        train = np.hstack(train)
        X_train = X[train]
        y_train = y_transformed[train]
        classifiers[a_idx] = classifiers[a_idx].fit(X_train, y_train)
        P_predict = classifiers[a_idx].predict_proba(X)
        cumlative = P_predict.cumsum(axis=1)
        uniform = random_state.rand(len(cumlative), 1)
        y_predict = (uniform < cumlative).argmax(axis=1)
        y[:, a_idx] = le.inverse_transform(y_predict)

    return y


def compute_annot_perf_clf(y_true, y, missing_label=-1):
    """
    Prints the performances of annotators for classification problems, i.e., micro and macro accuracies.

    Parameters
    ----------
    y_true: array-like of shape (n_samples)
        True class labels.
    y : array-like of shape (n_samples, n_annotators)
        Labels provided by the annotators.
    """
    y_true = column_or_1d(y_true)
    y = np.array(y)
    n_annotators = y.shape[1]
    acc = np.empty((2, n_annotators))
    for a in range(n_annotators):
        is_labeled_a = is_labeled(y[:, a], missing_label=missing_label)
        y_a = y[is_labeled_a, a]
        y_true_a = y_true[is_labeled_a]
        acc[0, a] = accuracy_score(y_true=y_true_a, y_pred=y_a)
        acc[1, a] = balanced_accuracy_score(y_true=y_true_a, y_pred=y_a)
    acc = pd.DataFrame(
        acc,
        index=["micro accuracy", "macro accuracy"],
        columns=np.arange(n_annotators),
    )
    return acc


def annot_sim_clf_cluster(
    X,
    y_true,
    cluster_annot_perfs,
    k_means_dict=None,
    random_state=None,
):
    """
    The knowledge of annotators is separated into clusters, where on each cluster an annotator can have different
    performances. These performances are expressed through labeling accuracies. The clusters are determined through a
    k-means algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Samples of the whole data set.
    y_true : array-like, shape (n_samples)
        True class labels of the given samples X.
    cluster_annot_perfs : array-like of shape (n_annotators, n_clusters)
        The entry `cluster_annot_perfs[j, i]` indicates the accuracy of annotator `j` for labeling samples of
        cluster `i`.
    k_means_dict : None or dict, optional (default=None)
        Dictionary of parameters that are passed to `sklearn.cluster.MiniBatchKMeans`.
    random_state : None or int or numpy.random.RandomState, optional (default=None)
        The random state used for drawing the annotations and specifying the clusters.

    Returns
    -------
    y : np.ndarray of shape (n_samples, n_annotators)
        Class labels of simulated annotators.
    """
    # Check `X` and `y_true`.
    X, y_true = check_X_y(X, y_true, ensure_2d=True, allow_nd=False)
    n_samples = X.shape[0]

    # Check `cluster_annot_perfs`.
    cluster_annot_perfs = check_array(cluster_annot_perfs)
    if np.sum(cluster_annot_perfs < 0) or np.sum(cluster_annot_perfs > 1):
        raise ValueError("`cluster_perfs` must contain values in [0, 1]")
    n_annotators = cluster_annot_perfs.shape[0]
    n_clusters = cluster_annot_perfs.shape[1]

    # Check `k_means_dict`.
    if k_means_dict is None:
        k_means_dict = {
            "batch_size": 2 ** 13,
            "random_state": random_state,
            "max_iter": 1000,
            "n_init": 10,
        }
    if not isinstance(k_means_dict, dict):
        raise TypeError("`k_means_dict` must be a dictionary.")

    # Check and transform `random_state`.
    random_state = check_random_state(random_state)

    # Transform class labels to interval [0, n_classes-1].
    le = LabelEncoder().fit(y_true)
    y_true = le.transform(y_true)
    n_classes = len(le.classes_)

    # Compute clustering.
    y_cluster = MiniBatchKMeans(n_clusters=n_clusters, **k_means_dict).fit_predict(X)

    # Simulate annotators.
    y = np.empty((n_samples, n_annotators))
    for a_idx in range(n_annotators):
        P_predict = np.empty((n_samples, n_classes))
        for c_idx in range(n_clusters):
            is_c = y_cluster == c_idx
            p = (1 - cluster_annot_perfs[a_idx, c_idx]) / (n_classes - 1)
            P_predict[is_c] = p
            P_predict[is_c, y_true[is_c]] = cluster_annot_perfs[a_idx, c_idx]
        cumlative = P_predict.cumsum(axis=1)
        uniform = random_state.rand(len(cumlative), 1)
        y_predict = (uniform < cumlative).argmax(axis=1)
        y[:, a_idx] = le.inverse_transform(y_predict)

    return y, y_cluster


def generate_expert_cluster_combinations(n_annotators, n_clusters, n_expert_clusters, random_state):
    """
    Helper function to randomly select expert clusters of annotators.

    Parameters
    ----------
    n_annotators : int
        Number of annotators.
    n_clusters : int
        Nuber of clusters.
    n_expert_clusters : int
        Number of expert clusters per annotator.
    random_state : int or np.random.RandomState or None, optional (default=None)
        Random state for selecting expert clusters.
    """
    combs = []
    combs_list = np.array(list(combinations(np.arange(n_clusters), n_expert_clusters)))
    random_state = check_random_state(random_state)
    random_order = random_state.choice(np.arange(len(combs_list)), size=len(combs_list), replace=False)
    combs_list = combs_list[random_order]
    while True:
        for comb in combs_list:
            combs.append(list(comb))
            if len(combs) == n_annotators:
                return np.array(combs, dtype=int)
