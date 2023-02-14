import numpy as np

from os.path import exists

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torchvision import transforms

# ADJUST absolut path to the data sets.
DATA_PATH = "./lfma_data_sets"
TABULAR_DATA_SETS = [
    "toy-classification",
    "letter",
    "label-me",
    "music",
]
BW_IMAGE_DATA_SETS = ["fmnist"]
RGB_IMAGE_DATA_SETS = ["cifar10", "svhn"]


def load_data(
    data_set_name,
    use_annotator_features=False,
    preprocess=True,
    data_path="./",
    valid_size=0.0,
    test_size=0.0,
    n_repeats=1,
    random_state=0,
    data_type="none",
):
    """Load data of given data set name.

    Parameters
    ----------
    data_set_name: str
        Name of the data set. The available data sets are list in `TABULAR_DATA_SETS`, `BW_IMAGE_DATASETS`,
        and `RGB_IMAGE_DATA_SETS`.
    use_annotator_features : bool, optional (default=False)
        Flag whether prior information about annotators should be used.
        Only returned, if this information is available for a given data set.
        Otherwise, annotators are represented through one-hot encoded vectors.
    preprocess : bool, optional (default=True)
        Flag whether the samples and annotator features are to be preprocessed.
    data_path : str, optional (default='./')
        Data path where data sets are stored.
    valid_size : float in (0, 1), optional (default=0.0)
        Ratio of validation samples.
    test_size : float in (0, 1), optional (default=0.0)
        Ratio of test samples.
    n_repeats : int, optional (default=1)
        This parameter is only important, if `test_size` or `valid_size` is greater zero.
        It determines the number of train, validation, and test splits.
    random_state : None or int or np.random.RandomState, optional (default=0)
        Random state to ensure reproducibility when splitting data into train, test, and validation sets.
    data_type : {"none", "rand-dep_x_y", "rand-indep_x_y", "inductive_x"}
        Determines the type of annotator simulation. "none" refers to the default case of simulated or existing
        real-world annotators. The other types refer to special annotator simulation settings:
        - "naive_x_y": There x "usual" annotators and "y" naive annotators assigning each sample to the same class.
        - "rand-dep_x_y": There x "usual" annotators and "y-x" copies of a randomly guessing annotator.
        - "rand-indep_x_y": There x "usual" annotators and "y-x" independent randomly guessing annotators.
        - "inductive_x": There 200-x "usual" annotators and "x" test annotators being not present during training.

    Returns
    -------
    X : numpy.ndarray, shape (n_samples, *)
        Samples whose shape depends on the data set itself.
    y_true : numpy.ndarray, shape (n_samples, )
        Ground truth labels of samples.
    y : numpy.ndarray, shape (n_samples, n_annotators)
        Annotation of each annotator for each sample.
    A : numpy.ndarray, shape (n_annotators, n_annot_features)
        Annotator features. If no explicit descriptions of the
        annotators exist, the annotators are represented as one-hot
        encoded vectors.
    """
    data_set_list = TABULAR_DATA_SETS + BW_IMAGE_DATA_SETS + RGB_IMAGE_DATA_SETS
    if data_set_name not in data_set_list:
        raise ValueError(f"Invalid data set. Only the following data sets are available: " f"{data_set_list}.")

    ds = {}
    data_set_path = f"{data_path}/{data_set_name}"
    ds["X"] = np.load(f"{data_set_path}-X.npy").astype(np.float32)
    ds["y_true"] = np.load(f"{data_set_path}-y-true.npy").astype(np.int)

    if data_type == "none":
        ds["y"] = np.load(f"{data_set_path}-y.npy").astype(np.int)
        annot_features_path = f"{data_set_path}-A.npy"
        if use_annotator_features and exists(annot_features_path):
            ds["A"] = np.load(annot_features_path).astype(np.float32)
            if preprocess:
                ds["A"] = StandardScaler().fit_transform(ds["A"])
        else:
            ds["A"] = np.eye(ds["y"].shape[1], dtype=np.float32)
    elif data_type == "correlated":
        y = np.load(f"{data_set_path}-y.npy").astype(np.int)
        indices = [0, 1, 9] * 10
        ds["y"] = np.column_stack((y, y[:, indices]))
        annot_features_path = f"{data_set_path}-A.npy"
        if use_annotator_features and exists(annot_features_path):
            A = np.load(annot_features_path).astype(np.float32)
            A = np.row_stack((A, A[indices]))
            if preprocess:
                ds["A"] = StandardScaler().fit_transform(A)
        else:
            ds["A"] = np.eye(ds["y"].shape[1], dtype=np.float32)
    elif "rand-indep" in data_type:
        splits = data_type.split("_")
        n_annotators = int(splits[1])
        n_random_annotators = int(splits[2]) - n_annotators
        ds["y"] = np.load(f"{data_set_path}-y-random.npy").astype(np.int)
        y_norm = ds["y"][:, :n_annotators]
        y_rand = ds["y"][:, n_annotators : n_annotators + n_random_annotators]
        ds["y"] = np.column_stack((y_norm, y_rand))
        annot_features_path = f"{data_set_path}-A-random.npy"
        if use_annotator_features and exists(annot_features_path):
            ds["A"] = np.load(annot_features_path).astype(np.float32)
            ds["A"] = np.row_stack(
                (
                    ds["A"][:n_annotators],
                    ds["A"][n_annotators : n_annotators + n_random_annotators],
                )
            )
            if preprocess:
                ds["A"] = StandardScaler().fit_transform(ds["A"])
        else:
            ds["A"] = np.eye(ds["y"].shape[1], dtype=np.float32)
    elif "rand-dep" in data_type:
        splits = data_type.split("_")
        n_annotators = int(splits[1])
        n_random_annotators = int(splits[2]) - n_annotators
        ds["y"] = np.load(f"{data_set_path}-y-random.npy").astype(np.int)
        y_norm = ds["y"][:, :n_annotators]
        y_rand = np.array([ds["y"][:, -1] for _ in range(n_random_annotators)])
        ds["y"] = np.column_stack((y_norm, y_rand.T))
        annot_features_path = f"{data_set_path}-A-random.npy"
        if use_annotator_features and exists(annot_features_path):
            ds["A"] = np.load(annot_features_path).astype(np.float32)
            A_norm = ds["A"][:n_annotators]
            A_random = np.array([ds["A"][-1] for _ in range(n_random_annotators)])
            ds["A"] = np.row_stack((A_norm, A_random))
            if preprocess:
                ds["A"] = StandardScaler().fit_transform(ds["A"])
        else:
            ds["A"] = np.eye(ds["y"].shape[1], dtype=np.float32)
    elif "naive" in data_type:
        splits = data_type.split("_")
        n_annotators = int(splits[1])
        n_naive_annotators = int(splits[2]) - n_annotators
        ds["y"] = np.load(f"{data_set_path}-y-naive.npy").astype(np.int)
        y_norm = ds["y"][:, :n_annotators]
        y_naive = np.array([ds["y"][:, -1] for _ in range(n_naive_annotators)])
        ds["y"] = np.column_stack((y_norm, y_naive.T))
        annot_features_path = f"{data_set_path}-A-naive.npy"
        if use_annotator_features and exists(annot_features_path):
            ds["A"] = np.load(annot_features_path).astype(np.float32)
            A_norm = ds["A"][:n_annotators]
            A_naive = np.array([ds["A"][-1] for _ in range(n_naive_annotators)])
            ds["A"] = np.row_stack((A_norm, A_naive))
            if preprocess:
                ds["A"] = StandardScaler().fit_transform(ds["A"])
        else:
            ds["A"] = np.eye(ds["y"].shape[1], dtype=np.float32)
    elif "inductive" in data_type:
        ds["y"] = np.load(f"{data_set_path}-y-inductive.npy").astype(np.int)
        if use_annotator_features:
            ds["A"] = np.load(f"{data_set_path}-A-inductive.npy").astype(np.float32)
            if preprocess:
                ds["A"] = StandardScaler().fit_transform(ds["A"])
        else:
            ds["A"] = np.eye(ds["y"].shape[1], dtype=np.float32)

    if data_set_name in ["label-me", "music"]:
        # Define constant train indices.
        train = np.arange(len(ds["X"]))
        ds["train"] = np.array([train for _ in range(n_repeats)], dtype=int)

        # Load test data.
        X_test = np.load(f"{data_set_path}-X-test.npy")
        y_true_test = np.load(f"{data_set_path}-y-true-test.npy")
        y_test = np.full((len(X_test), ds["y"].shape[1]), fill_value=-1)

        # Define constant test indices.
        test = np.arange(len(train), len(train) + len(X_test))
        ds["test"] = np.array([test for _ in range(n_repeats)], dtype=int)

        if valid_size > 0:
            # Load validation data.
            X_valid = np.load(f"{data_set_path}-X-valid.npy")
            y_true_valid = np.load(f"{data_set_path}-y-true-valid.npy")
            y_valid = np.full((len(X_valid), ds["y"].shape[1]), fill_value=-1)

            # Add validation data.
            ds["X"] = np.vstack((ds["X"], X_valid))
            ds["y_true"] = np.concatenate((ds["y_true"], y_true_valid))
            ds["y"] = np.vstack((ds["y"], y_valid))

            valid = np.arange(len(X_valid)) + len(train)
            ds["valid"] = np.array([valid for _ in range(n_repeats)], dtype=int)
            ds["test"] += len(X_valid)

        # Add test data.
        ds["X"] = np.vstack((ds["X"], X_test))
        ds["y_true"] = np.concatenate((ds["y_true"], y_true_test))
        ds["y"] = np.vstack((ds["y"], y_test))
    else:
        train_indices = []
        test_indices = []
        valid_indices = [] if valid_size > 0 else None
        for r in range(n_repeats):
            train, test = train_test_split(
                np.arange(len(ds["X"])),
                test_size=test_size,
                random_state=random_state,
            )
            if valid_indices is not None:
                train, valid = train_test_split(
                    train, test_size=valid_size / (1 - test_size), random_state=random_state
                )
                valid_indices.append(valid)
            train_indices.append(train)
            test_indices.append(test)
        ds["train"] = np.array(train_indices, dtype=int)
        ds["test"] = np.array(test_indices, dtype=int)
        if valid_indices is not None:
            ds["valid"] = np.array(valid_indices, dtype=int)
    ds["y_true_ap"] = ds["y_true"][:, np.newaxis] == ds["y"]
    ds["y_true_ap"] = ds["y_true_ap"].astype(int)
    ds["y_true_ap"][ds["y"] == -1] = -1

    if preprocess:
        if data_set_name in ["label-me"]:
            pass
        elif data_set_name in TABULAR_DATA_SETS:
            ds["X"] = StandardScaler().fit_transform(ds["X"])
        elif data_set_name in RGB_IMAGE_DATA_SETS + BW_IMAGE_DATA_SETS:
            ds["X"] /= 255
            ds["X"] -= np.mean(ds["X"], axis=(0, 2, 3), keepdims=True)
            ds["X"] /= np.std(ds["X"], axis=(0, 2, 3), keepdims=True)

    if data_set_name == "fmnist":
        ds["transform"] = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(0.15),
                transforms.RandomCrop(28, padding=4),
            ]
        )
    elif data_set_name == "cifar10":
        ds["transform"] = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif data_set_name == "svhn":
        ds["transform"] = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
            ]
        )
    else:
        ds["transform"] = None

    return ds
