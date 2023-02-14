import numpy as np
import os
import pandas as pd

from copy import deepcopy

from inspect import signature

from lfma.utils import introduce_missing_annotations, LitProgressBar, StoreBestModuleStateDict

from sacred import Experiment

from evaluation.data_utils import (
    load_data,
    DATA_PATH,
)
from evaluation.experiment_utils import (
    get_perf_metrics,
    aggregate_labels,
    instantiate_model,
)

from evaluation.config_utils import config_dict_to_file_name

from pytorch_lightning import seed_everything

# ADJUST the absolute path where the results are to be saved.
RESULT_PATH = "./lfma_results"

ex = Experiment()


@ex.config
def config():
    # Random seed for reproducibility.
    seed = 0

    # Further data set names are list in `evaluation/data_utils.py`.
    data_set_name = "letter"

    # Corresponds to the annotator sets in the accompanied article, where we have the following mapping:
    # none=independent, correlated=interdependent, rand_dep_10_100=random-interdependent, and inductive=inductive_25.
    data_type = "none"

    # Number of repeated experiments, i.e., train, validation, and test splits.
    n_repeats = 5

    # Fraction of test samples, which is only relevant for data sets without predefined splits.
    test_size = 0.2

    # Fraction of validation samples, which is only relevant for data sets without predefined splits.
    valid_size = 0.05

    # Fraction of missing annotations.
    missing_label_ratio = 0.8

    # Parameters passed to the `pytorch_lightning.Trainer` object.
    trainer_dict = {
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": 1,
        "enable_progress_bar": True,
        "logger": False,
        "enable_checkpointing": False,
    }

    # Name of the possible optimizer. See `evaluation/experiment_utils.py` for details.
    optimizer = "AdamW"

    # Parameters passed to the `torch.optim.Optimizer` object.
    optimizer_dict = {"lr": 0.01, "weight_decay": 0.0}

    # Name of the possible learning rate schedulers. See `evaluation/experiment_utils.py` for details.
    lr_scheduler = "CosineAnnealing"

    # Parameters passed to the `torch.optim.lr_scheduler.LRScheduler` object.
    lr_scheduler_dict = None

    # Batch sized used during training.
    batch_size = 64

    # Dropout rate used during training.
    dropout_rate = 0.0

    # Name of the multi-annotator learning technique. See `evaluation/experiment_utils.py` for details.
    model_name = "cl"

    # Parameters passed to the module of the multi-annotator learning technique. See the respective moudle class
    # in `lfma/modules` for details.
    model_dict = {}


@ex.automain
def run_lfma_experiment(
    data_set_name,
    data_type,
    n_repeats,
    valid_size,
    test_size,
    missing_label_ratio,
    trainer_dict,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    batch_size,
    dropout_rate,
    model_name,
    model_dict,
    seed,
):
    # Get configuration and set filename.
    config_str = config_dict_to_file_name(ex.current_run.config)
    print(config_str)
    result_dir = f"{RESULT_PATH}/{data_set_name}/{data_type}"
    if not os.path.exists(result_dir):
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            print(f"Directory {result_dir} already existed. Continue.")
    filename = f"{result_dir}/{config_str}.csv"

    # Check whether file already exists.
    if os.path.isfile(filename):
        return 0

    # Copy configuration.
    optimizer_dict = deepcopy(optimizer_dict)
    model_dict = deepcopy(model_dict)

    # Load data.
    use_annotator_features = model_dict.pop("use_annotator_features", False)
    ds = load_data(
        data_path=DATA_PATH,
        data_set_name=data_set_name,
        data_type=data_type,
        use_annotator_features=use_annotator_features,
        preprocess=True,
        n_repeats=n_repeats,
        valid_size=valid_size,
        test_size=test_size,
    )

    # Get performance metrics according to learning task.
    (
        perf_metrics_gt,
        perf_metrics_ap,
        perf_dict,
    ) = get_perf_metrics(data_types=["train", "valid", "test"], inductive="inductive" in data_type)

    n_iter = 0
    for tr, val, te in zip(ds["train"], ds["valid"], ds["test"]):
        n_iter += 1
        print(f"Train {model_name} on {n_iter}-th fold of {data_set_name}.")

        # Add callbacks.
        td = deepcopy(trainer_dict)
        td["callbacks"] = [StoreBestModuleStateDict(score_name="val_acc", maximize=True)]
        if trainer_dict["enable_progress_bar"]:
            td["callbacks"].append(LitProgressBar())

        # Set global random seed.
        seed_everything(seed + n_iter)

        # Randomly add missing labels.
        y_partial = introduce_missing_annotations(
            y=ds["y"][tr],
            missing_label=-1,
            percentage=missing_label_ratio,
            random_state=seed + n_iter,
        )
        test_annot_indices = np.arange(ds["y"].shape[1])
        if "inductive" in data_type:
            splits = data_type.split("_")
            n_annotators = ds["y"].shape[1]
            n_test_annot = int(splits[1])
            inductive_annot_indices = np.arange(0, n_annotators, n_annotators // n_test_annot).astype(int)
            y_partial[:, inductive_annot_indices] = -1
            test_annot_indices = np.setdiff1d(test_annot_indices, inductive_annot_indices)

        # Generate model according to configuration.
        model = instantiate_model(
            data_set_name=data_set_name,
            data_set=ds,
            model_name=model_name,
            trainer_dict=deepcopy(td),
            optimizer=optimizer,
            optimizer_dict=deepcopy(optimizer_dict),
            lr_scheduler=lr_scheduler,
            lr_scheduler_dict=deepcopy(lr_scheduler_dict),
            dropout_rate=dropout_rate,
            model_dict=deepcopy(model_dict),
            random_state=seed + n_iter,
        )

        # Fit model on training data.
        data_loader_dict = {"batch_size": batch_size, "shuffle": True, "drop_last": True}
        val_data_loader_dict = {"batch_size": 256, "shuffle": False, "drop_last": False}
        fit_dict = {
            "X": ds["X"][tr],
            "y": y_partial[:, test_annot_indices],
            "X_val": ds["X"][val],
            "y_val": ds["y_true"][val],
            "data_loader_dict": data_loader_dict,
            "val_data_loader_dict": val_data_loader_dict,
            "transform": ds["transform"],
            "val_transform": None,
        }
        if model_name in ["madl", "conal", "gt", "mr", "lia"]:
            fit_dict["A"] = ds["A"][test_annot_indices]
        if model_name in ["gt", "mr"]:
            fit_dict["y_agg"] = aggregate_labels(
                y=y_partial,
                y_true=ds["y_true"][tr],
                aggregation_method=model_name,
            )
        model.fit(**fit_dict)

        # Load model with the highest validation accuracy measured after each training epoch.
        best_model_state_dict = model.trainer_dict_["callbacks"][0].best_model_state_dict
        model.module_.load_state_dict(state_dict=best_model_state_dict)

        # Obtain probabilistic and categorical ground truth predictions for training, validation, and test instances.
        y_proba = model.predict_proba(X=ds["X"], data_loader_dict=val_data_loader_dict)
        y_pred = np.argmax(y_proba, axis=1)

        # Evaluate ground truth predictions on train, validation, and test data.
        for subset, idx in zip(["train", "valid", "test"], [tr, val, te]):
            for name, metric_info in perf_metrics_gt.items():
                metric, is_probabilistic = metric_info[0], metric_info[1]
                y_hat = y_proba if is_probabilistic else y_pred
                score = metric(y_true=ds["y_true"][idx], y_pred=y_hat[idx])
                perf_dict[f"{subset}-{name}"].append(score)
                print(f"GT {subset}-{name}: {score:.3f}")

        # Obtain probabilistic and categorical annotator performance predictions for the annotators who were available
        # during training (transductive).
        y_proba = model.predict_annotator_perf(X=ds["X"], data_loader_dict=val_data_loader_dict)
        y_proba = np.stack((1 - y_proba, y_proba), axis=2)
        y_pred = np.argmax(y_proba, axis=2)

        # Obtain probabilistic and categorical annotator performance predictions for the annotators who were not
        # available during training (inductive).
        y_proba_inductive = None
        y_pred_inductive = None
        model_is_inductive = "A" in signature(model.predict_annotator_perf).parameters
        if "inductive" in data_type and use_annotator_features and model_is_inductive:
            y_proba_inductive = model.predict_annotator_perf(
                X=ds["X"], A=ds["A"][inductive_annot_indices], data_loader_dict=val_data_loader_dict
            )
            y_proba_inductive = np.stack((1 - y_proba_inductive, y_proba_inductive), axis=2)
            y_pred_inductive = np.argmax(y_proba_inductive, axis=2)

        # Evaluate annotator performance predictions on train, validation, and test data.
        for subset, idx in zip(["train", "valid", "test"], [tr, val, te]):
            for name, metric_info in perf_metrics_ap.items():
                metric, is_probabilistic = metric_info[0], metric_info[1]
                if is_probabilistic:
                    y_hat = y_proba[idx]
                else:
                    y_hat = y_pred[idx]
                if np.any(ds["y_true_ap"] == -1):
                    score = np.nan
                else:
                    score = metric(y_true=ds["y_true_ap"][idx][:, test_annot_indices], y_pred=y_hat)
                perf_dict[f"{subset}-{name}"].append(score)
                print(f"AP {subset}-{name}: {score:.3f}")
                if "inductive" in data_type:
                    y_hat = None
                    if y_proba_inductive is not None and y_pred_inductive is not None:
                        if is_probabilistic:
                            y_hat = y_proba_inductive[idx]
                        else:
                            y_hat = y_pred_inductive[idx]
                    if np.any(ds["y_true_ap"] == -1) or y_hat is None:
                        score = np.nan
                    else:
                        score = metric(y_true=ds["y_true_ap"][idx][:, inductive_annot_indices], y_pred=y_hat)
                    perf_dict[f"{subset}-{name}-inductive"].append(score)
                    print(f"AP {subset}-{name}-inductive: {score:.3f}")

    # Store performances as artifacts.
    perf_df = pd.DataFrame(perf_dict)
    perf_df.to_csv(path_or_buf=filename, index=False)
    print(perf_df.mean(axis=0))
