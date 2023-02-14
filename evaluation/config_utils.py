import glob
import numpy as np
import pandas as pd

from numpy.core.defchararray import find

# Dictionary of abbreviations for saving the results of experimental configurations.
CONFIG_ABBREVIATIONS = {
    "data_set_name": "ds",
    "data_type": "dt",
    "model_name": "mdl",
    "missing_label_ratio": "mlr",
    "n_repeats": "nr",
    "test_size": "ts",
    "valid_size": "vs",
    "max_epochs": "me",
    "accelerator": "dev",
    "confusion_matrix": "cm",
    "batch_size": "bs",
    "ap_sim_func": "sim",
    "ap_embed_scaling_exponent": "exp",
    "ap_embed_scaling_base": "base",
    "ap_use_residual": "res",
    "use_annotator_features": "uaf",
    "aggregation_method": "agg",
    "model_params": "mp",
    "embed_size": "es",
    "learned": "lea",
    "dropout_rate": "dr",
    "ap_use_outer_product": "op",
    "enable_progress_bar": "pb",
    "logger": "lg",
    "lr_scheduler": "ls",
    "CosineAnnealing": "ca",
    "CosineAnnealingRestarts": "car",
    "MultiStepLR": "ms",
    "optimizer": "opt",
    "Adam": "ad",
    "AdamW": "adw",
    "SGD": "sgd",
    "nesterov": "nv",
    "momentum": "mom",
    "True": "+",
    "False": "-",
    "weight_decay": "wd",
    "ap_latent_dim": "ald",
    "n_fine_tune_epochs": "nfte",
    "n_em_steps": "nes",
}


def config_abbreviate(config_param):
    """
    Abbreviate given parameter using the `CONFIG_ABBREVIATION` dictionary.

    Parameters
    ----------
    config_param : str
        Parameter to be abbreviated.

    Returns
    -------
    config_param: str
        If an entry for an abbreviation exists, the abbreviated parameter is returned, otherwise the original parameter
        is returned.
    """
    if config_param in CONFIG_ABBREVIATIONS.keys():
        return CONFIG_ABBREVIATIONS[config_param]
    return config_param


def config_dict_to_file_name(config_dict, init=True):
    """
    Transforms a given configuration as dictionary into a string, which can be used as file name.

    Parameters
    ----------
    config_dict : dict
        Dictionary of the configuration to be transformed.
    init : bool, optional (default=True)
        Flag indicating whether this is the initial call (`init=True`) or a recursive call (`init=False`) of this
        function.

    Returns
    -------
    name : str
        Configuration transformed to a string.
    """
    name = ""
    if init:
        name += f"|{config_abbreviate(config_dict.pop('seed'))}|"
    for idx, key_item in enumerate(config_dict.items()):
        key, item = key_item[0], key_item[1]
        if key in ["logger", "enable_progress_bar", "enable_checkpointing", "devices", "dev"]:
            continue
        if isinstance(item, dict):
            name += config_dict_to_file_name(item, init=False)
        else:
            key = config_abbreviate(key)
            item = config_abbreviate(str(item))
            name += f"{key}={item}|"
    return name


def config_query(config_dict, path="./"):
    """
    Returns a dictionary of data frames (items) whose names (keys) match the parameters in the configuration
    dictionary.

    Parameters
    ----------
    config_dict : dict
        Dictionary defining a query to find matching files.
    path : str
        Path to files.

    Returns
    -------
    res_dict : dict
        Dictionary consisting of matched results as data frames (items) with file names (keys).
    """
    ds = config_dict["data_set_name"]
    dt = config_dict["data_type"]
    res = np.array(glob.glob(f"{path}/{ds}/{dt}/*.csv"))
    for key, item in config_dict.items():
        key = config_abbreviate(str(key))
        item = config_abbreviate(str(item))
        res = np.concatenate((res[find(res, f"|{key}={item}.csv") != -1], res[find(res, f"|{key}={item}|") != -1]))
    res_dict = {}
    for file in res:
        res_dict[file] = pd.read_csv(file, index_col=False)
    return res_dict


def gen_results_table(res_dict, param):
    """
    Takes as input a dictionary of results, provided by the `config_query` function, and computes the mean results
    for the different values of the given parameter.

    Parameters
    ----------
    res_dict : dict
        Dictionary of results provided by the `config_query` function.
    param : str
        Parameter whose impact of the model performance for different values is to be studied.

    Returns
    -------
    res_df : pd.DataFrame
        Date frame with results structured according to the different values of `param`.
    """
    res_table = {}
    for key, item in res_dict.items():
        parts = key.replace(".csv", "").split("|")[1:]
        for part_idx, part in enumerate(parts):
            sub_parts = part.split("=")
            if sub_parts[0] == config_abbreviate(param):
                param_value = sub_parts[1]
                if param == "alpha":
                    beta_part = [p for p in parts if "beta" in p][0]
                    param_value = f"{param_value}_{beta_part.split('=')[1]}"
                if param_value not in res_table:
                    res_table[param_value] = {}
                    res_table[param_value]["valid-micro-accuracy-gt"] = np.array([-np.inf] * len(item))
        is_better = item["valid-micro-accuracy-gt"] > res_table[param_value]["valid-micro-accuracy-gt"]
        for perf_key in item:
            if not perf_key in res_table[param_value]:
                res_table[param_value][perf_key] = item[perf_key]
            else:
                res_table[param_value][perf_key][is_better] = item[perf_key][is_better]
    res_df = {"param": [], "value": []}
    for param_value, param_value_dict in res_table.items():
        if param == "alpha":
            res_df["param"].append("alpha_beta")
        else:
            res_df["param"].append(param)
        res_df["value"].append(param_value)
        for perf_key, perf_scores in param_value_dict.items():
            if not perf_key in res_df:
                res_df[perf_key] = [f"{np.round(perf_scores.mean(), 3):.3f} +- {np.round(perf_scores.std(), 3):.3f}"]
            else:
                res_df[perf_key].append(
                    f"{np.round(perf_scores.mean(), 3):.3f} +- {np.round(perf_scores.std(), 3):.3f}"
                )
    res_df = pd.DataFrame(res_df)
    return res_df
