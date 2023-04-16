import numpy as np

from lfma.utils import brier_score, cross_entropy_score, micro_accuracy_score, macro_accuracy_score

from skactiveml.utils import majority_vote

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, CosineAnnealingWarmRestarts

from evaluation.architecture_utils import (
    instantiate_madl_classifier,
    instantiate_crowd_layer_classifier,
    instantiate_conal_classifier,
    instantiate_aggregate_classifier,
    instantiate_union_net_classifier,
    instantiate_reac_classifier,
    instantiate_lia_classifier,
)


def get_perf_metrics(data_types, inductive=False):
    # Setup metric dictionaries.
    perf_metrics_gt = {
        "micro-accuracy-gt": [micro_accuracy_score, False],
        "macro-accuracy-gt": [macro_accuracy_score, False],
        "cross-entropy-gt": [cross_entropy_score, True],
        "brier-score-gt": [brier_score, True],
    }
    perf_metrics_ap = {
        "micro-accuracy-ap": [micro_accuracy_score, False],
        "macro-accuracy-ap": [macro_accuracy_score, False],
        "cross-entropy-ap": [cross_entropy_score, True],
        "brier-score-ap": [brier_score, True],
    }

    # Setup dictionaries for the resulting performances.
    perf_dict = {}
    for d in data_types:
        for key in perf_metrics_gt.keys() | perf_metrics_ap.keys():
            perf_dict[f"{d}-{key}"] = []
    if inductive:
        for d in data_types:
            for key in perf_metrics_ap.keys():
                perf_dict[f"{d}-{key}-inductive"] = []

    return perf_metrics_gt, perf_metrics_ap, perf_dict


def aggregate_labels(y, y_true, aggregation_method):
    if aggregation_method == "gt":
        return y_true
    elif aggregation_method == "mr":
        return majority_vote(y=y, missing_label=-1)
    else:
        raise ValueError('`aggregation_method` must be in `["mr", "gt"]`.')


def instantiate_model(
    data_set_name,
    data_set,
    model_name,
    trainer_dict,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    dropout_rate,
    model_dict,
    random_state,
):
    classes = np.unique(data_set["y_true"])
    n_features = data_set["X"].shape[1]
    n_ap_features = data_set["A"].shape[1]

    model_dict["data_set_name"] = data_set_name
    model_dict["classes"] = classes
    model_dict["n_features"] = n_features
    model_dict["trainer_dict"] = trainer_dict
    model_dict["dropout_rate"] = dropout_rate
    model_dict["missing_label"] = -1
    model_dict["random_state"] = random_state

    # Set optimizer.
    if optimizer == "Adam":
        model_dict["optimizer"] = Adam
    elif optimizer == "AdamW":
        model_dict["optimizer"] = AdamW
    elif optimizer == "SGD":
        model_dict["optimizer"] = SGD
    elif optimizer is None:
        model_dict["optimizer"] = None
    else:
        raise ValueError("`optimizer` needs to be in ['Adam', 'AdamW', 'SGD']")
    model_dict["optimizer_dict"] = optimizer_dict

    # Set learning rate scheduler.
    if lr_scheduler == "CosineAnnealing":
        model_dict["lr_scheduler"] = CosineAnnealingLR
    elif lr_scheduler == "CosineAnnealingRestarts":
        model_dict["lr_scheduler"] = CosineAnnealingWarmRestarts
    elif lr_scheduler == "MultiStepLR":
        model_dict["lr_scheduler"] = MultiStepLR
    elif lr_scheduler is None:
        model_dict["lr_scheduler"] = None
    else:
        raise ValueError("`lr_scheduler` needs to be in ['CosineAnnealing', 'MultiStepLR', CosineAnnealingRestarts]")
    model_dict["lr_scheduler_dict"] = lr_scheduler_dict

    if model_name == "madl":
        model = instantiate_madl_classifier(n_ap_features=n_ap_features, **model_dict)
    elif model_name == "cl":
        model = instantiate_crowd_layer_classifier(**model_dict)
    elif model_name == "union":
        model = instantiate_union_net_classifier(**model_dict)
    elif model_name == "reac":
        model = instantiate_reac_classifier(**model_dict)
    elif model_name == "conal":
        model = instantiate_conal_classifier(n_ap_features=n_ap_features, **model_dict)
    elif model_name == "lia":
        model = instantiate_lia_classifier(n_ap_features=n_ap_features, **model_dict)
    elif model_name in ["gt", "mr"]:
        model = instantiate_aggregate_classifier(n_ap_features=n_ap_features, **model_dict)
    else:
        raise ValueError(f"`model_name={model_name}` is an invalid parameter.")

    return model


def write_commands(
    file_name,
    commands,
    model_name,
    data_type,
    data_set_name,
    mem,
    n_parallel_jobs,
    cpus_per_task,
    slurm_logs_path,
    slurm_error_logs_path,
    use_slurm=True,
    use_gpu=False,
):
    commands = list(dict.fromkeys(commands).keys())
    n_tasks = len(commands)
    n_lines = 15 if use_gpu else 14
    if n_parallel_jobs > n_tasks:
        n_parallel_jobs = n_tasks
    print(f"{data_set_name}-{data_type}: {n_tasks}")
    sbatch_config = [
        f"#!/usr/bin/env bash",
        f"#SBATCH --job-name={model_name}_{data_set_name}_{data_type}",
        f"#SBATCH --array=1-{n_tasks}%{n_parallel_jobs}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --ntasks=1",
        f"#SBATCH --get-user-env",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --partition=main",
        f"#SBATCH --exclude=radagast,irmo,alatar",
        f"#SBATCH --output={slurm_logs_path}/{model_name}_{data_set_name}_{data_type}_%A_%a.log",
        f"#SBATCH --error={slurm_error_logs_path}/{model_name}_{data_set_name}_{data_type}_%A_%a.log",
    ]
    if use_gpu:
        sbatch_config.append("#SBATCH --gres=gpu:1")
    sbatch_config.extend(
        [
            f'eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+{n_lines})) p" {file_name})"',
            f"exit 0",
        ]
    )
    if not use_slurm:
        sbatch_config = [sbatch_config[0]]
    commands = sbatch_config + [""] + commands
    with open(file_name, "w") as f:
        for item in commands:
            f.write("%s\n" % item)
