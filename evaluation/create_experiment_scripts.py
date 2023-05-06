import os

from experiment_utils import write_commands

# Path to `run_experiment.py` Python script.
run_experiment_path = "/mnt/work/madl/evaluation/run_experiment.py"

# Flag whether the commands should be generated for a SLURM cluster.
use_slurm = True
python_command = "srun python -u" if use_slurm else "python -u"

# If slurm is available, this path defines where the log files are to be stored.
slurm_logs_path = "/mnt/work/madl/lfma_logs"
slurm_error_logs_path = "/mnt/work/madl/lfma_error_logs"

# General experimental setup.
data_set_name_list = ["letter", "fmnist", "cifar10", "svhn", "label-me"]
data_type_dict = {
    "letter": ["none", "rand-dep_10_100", "rand-indep_10_100", "correlated", "inductive_25"],
    "fmnist": ["none", "rand-dep_10_100", "rand-indep_10_100", "correlated", "inductive_25"],
    "cifar10": ["none", "rand-dep_10_100", "rand-indep_10_100", "correlated", "inductive_25"],
    "svhn": ["none", "rand-dep_10_100", "rand-indep_10_100", "correlated", "inductive_25"],
    "label-me": ["none"],
    "music": ["none"],
}
seed = 0
n_repeats = 5
test_size = 0.2
valid_size = 0.05
max_epochs = 100
default_dict = {"none": 0.8, "rand-dep_10_100": 0.8, "rand-indep_10_100": 0.8, "correlated": 0.8, "inductive_25": 0.98}
missing_label_dict = {
    "letter": default_dict,
    "fmnist": default_dict,
    "cifar10": default_dict,
    "svhn": default_dict,
    "label-me": {"none": 0.0},
    "music": {"none": 0.0},
}
drop_out_rate_dict = {
    "letter": 0.0,
    "fmnist": 0.0,
    "cifar10": 0.0,
    "svhn": 0.0,
    "label-me": 0.5,
    "music": 0.5,
}
batch_size_dict = {
    "letter": [64],
    "fmnist": [64],
    "cifar10": [64],
    "svhn": [64],
    "label-me": [64, 16, 8],
    "music": [64, 16, 8],
}
optimizer = "AdamW"
lr_list = [0.01, 0.005, 0.001]
weight_decay_list = [0.0001, 0.001, 0]
lr_scheduler = "CosineAnnealing"

# Server setup: Adjust the parameters according to your individual requirements.
mem_dict = {
    "none": "20gb",
    "rand-indep_10_100": "40gb",
    "rand-dep_10_100": "40gb",
    "correlated": "20gb",
    "inductive_25": "40gb",
}
parallel_jobs_dict = {
    "none": 100,
    "rand-indep_10_100": 50,
    "rand-dep_10_100": 50,
    "correlated": 100,
    "inductive_25": 50,
}
accelerator_dict = {
    "letter": "cpu",
    "fmnist": "cpu",
    "cifar10": "gpu",
    "svhn": "gpu",
    "label-me": "cpu",
    "music": "cpu",
}
devices_dict = {
    "letter": None,
    "fmnist": None,
    "cifar10": 1,
    "svhn": 1,
    "label-me": None,
    "music": None,
}
logger = False
cpus_per_task = 4
enable_progress_bar = False
enable_checkpointing = False

# ================================================ MaDL parameters ====================================================
model_name = "madl"
print("\n\n" + model_name)
batch_path = f"./{model_name}"
if not os.path.exists(batch_path):
    os.makedirs(batch_path)

embed_x_dict = {
    "none": ["none", "learned"],
    "rand-dep_10_100": ["learned"],
    "rand-indep_10_100": ["learned"],
    "correlated": ["learned"],
    "inductive_25": ["learned"],
}
confusion_matrix_dict = {
    "none": {"none": ["full", "diagonal", "isotropic"], "learned": ["full", "diagonal", "isotropic"]},
    "rand-dep_10_100": {"learned": ["full"]},
    "rand-indep_10_100": {"learned": ["full"]},
    "correlated": {"learned": ["full"]},
    "inductive_25": {"learned": ["full"]},
}
use_annotator_features_dict = {
    "none": [False],
    "rand-dep_10_100": [False],
    "rand-indep_10_100": [False],
    "correlated": [False],
    "inductive_25": [True, False],
}
ovat_dict = {
    "embed_size": [16, 8, 32],
    "eta": [0.8, 0.9, 0.7, 0.1],
    "alpha_beta": [
        [1.25, 0.25],
        [1.5, 0.5],
        [1.118, 0.236],
        [1.226, 0.452],
        [1.56, 0.28],
        [2.22, 0.61],
        [None, None]
    ],
    "ap_use_residual": {"none": [False], "learned": [True, False]},
    "ap_use_outer_product": {"none": [False], "learned": [True, False]},
}

for data_set_name, data_type_list in data_type_dict.items():
    for data_type in data_type_list:
        commands = []
        file_name = f"{batch_path}/{model_name}_{data_set_name}_{data_type}.sh"
        for lr in lr_list:
            for weight_decay in weight_decay_list:
                for batch_size in batch_size_dict[data_set_name]:
                    for embed_x in embed_x_dict[data_type]:
                        for confusion_matrix in confusion_matrix_dict[data_type][embed_x]:
                            if embed_x == "learned" and confusion_matrix == "full":
                                use_annotator_feature_list = use_annotator_features_dict[data_type]
                            else:
                                use_annotator_feature_list = [False]
                            for use_annotator_features in use_annotator_feature_list:
                                for param, value_list in ovat_dict.items():
                                    if isinstance(value_list, dict):
                                        value_list = value_list[embed_x]
                                    for value in value_list:
                                        commands.append(
                                            f"{python_command} {run_experiment_path} with "
                                            f"seed={seed} "
                                            f"data_set_name={data_set_name} "
                                            f"data_type={data_type} "
                                            f"n_repeats={n_repeats} "
                                            f"test_size={test_size} "
                                            f"valid_size={valid_size} "
                                            f"missing_label_ratio={missing_label_dict[data_set_name][data_type]} "
                                            f"trainer_dict.max_epochs={max_epochs} "
                                            f"trainer_dict.enable_progress_bar={enable_progress_bar} "
                                            f"trainer_dict.enable_checkpointing={enable_checkpointing} "
                                            f"trainer_dict.accelerator={accelerator_dict[data_set_name]} "
                                            f"trainer_dict.devices={devices_dict[data_set_name]} "
                                            f"trainer_dict.logger={logger} "
                                            f"optimizer={optimizer} "
                                            f"optimizer_dict.lr={lr} "
                                            f"optimizer_dict.weight_decay={weight_decay} "
                                            f"lr_scheduler={lr_scheduler} "
                                            f"lr_scheduler_dict.T_max={max_epochs} "
                                            f"batch_size={batch_size} "
                                            f"dropout_rate={drop_out_rate_dict[data_set_name]} "
                                            f"model_name={model_name} "
                                            f"model_dict.eta={ovat_dict['eta'][0]} "
                                            f"model_dict.confusion_matrix={confusion_matrix} "
                                            f"model_dict.alpha={ovat_dict['alpha_beta'][0][0]} "
                                            f"model_dict.beta={ovat_dict['alpha_beta'][0][1]} "
                                            f"model_dict.embed_size={ovat_dict['embed_size'][0]} "
                                            f"model_dict.embed_x={embed_x} "
                                            f"model_dict.use_annotator_features={use_annotator_features} "
                                            f"model_dict.ap_use_residual={ovat_dict['ap_use_residual'][embed_x][0]} "
                                            f"model_dict.ap_use_outer_product"
                                            f"={ovat_dict['ap_use_outer_product'][embed_x][0]}"
                                        )
                                        if (
                                            data_set_name in ["letter", "music", "label-me"]
                                            and embed_x == "learned"
                                            and confusion_matrix == "full"
                                        ):
                                            if param == "alpha_beta":
                                                commands[-1] = commands[-1].replace(
                                                    f"alpha={value_list[0][0]}", f"alpha={value[0]}"
                                                )
                                                commands[-1] = commands[-1].replace(
                                                    f"beta={value_list[0][1]}", f"beta={value[1]}"
                                                )
                                            else:
                                                commands[-1] = commands[-1].replace(
                                                    f"{param}={value_list[0]}", f"{param}={value}"
                                                )
                                        elif data_type in ["correlated", "rand-dep_10_100", "rand-indep_10_100"]:
                                            if param == "alpha_beta" and value[0] in [None, 1.25]:
                                                commands[-1] = commands[-1].replace(
                                                    f"alpha={value_list[0][0]}", f"alpha={value[0]}"
                                                )
                                                commands[-1] = commands[-1].replace(
                                                    f"beta={value_list[0][1]}", f"beta={value[1]}"
                                                )

        write_commands(
            file_name=file_name,
            commands=commands,
            model_name=model_name,
            data_type=data_type,
            data_set_name=data_set_name,
            mem=mem_dict[data_type],
            n_parallel_jobs=parallel_jobs_dict[data_type],
            cpus_per_task=cpus_per_task,
            slurm_logs_path=slurm_logs_path,
            slurm_error_logs_path=slurm_error_logs_path,
            use_slurm=use_slurm,
            use_gpu=accelerator_dict[data_set_name] == "gpu",
        )

# ================================================ CoNAL parameters ===================================================
model_name = "conal"
print("\n\n" + model_name)
batch_path = f"./{model_name}"
if not os.path.exists(batch_path):
    os.makedirs(batch_path)

embed_size = 20
lmbda = 0.00001

for data_set_name, data_type_list in data_type_dict.items():
    for data_type in data_type_list:
        commands = []
        file_name = f"{batch_path}/{model_name}_{data_set_name}_{data_type}.sh"
        for lr in lr_list:
            for weight_decay in weight_decay_list:
                for batch_size in batch_size_dict[data_set_name]:
                    for use_annotator_features in use_annotator_features_dict[data_type]:
                        commands.append(
                            f"{python_command} {run_experiment_path} with "
                            f"seed={seed} "
                            f"data_set_name={data_set_name} "
                            f"data_type={data_type} "
                            f"n_repeats={n_repeats} "
                            f"test_size={test_size} "
                            f"valid_size={valid_size} "
                            f"missing_label_ratio={missing_label_dict[data_set_name][data_type]} "
                            f"trainer_dict.max_epochs={max_epochs} "
                            f"trainer_dict.enable_progress_bar={enable_progress_bar} "
                            f"trainer_dict.enable_checkpointing={enable_checkpointing} "
                            f"trainer_dict.accelerator={accelerator_dict[data_set_name]} "
                            f"trainer_dict.devices={devices_dict[data_set_name]} "
                            f"trainer_dict.logger={logger} "
                            f"optimizer={optimizer} "
                            f"optimizer_dict.lr={lr} "
                            f"optimizer_dict.weight_decay={weight_decay} "
                            f"lr_scheduler={lr_scheduler} "
                            f"lr_scheduler_dict.T_max={max_epochs} "
                            f"batch_size={batch_size} "
                            f"dropout_rate={drop_out_rate_dict[data_set_name]} "
                            f"model_name={model_name} "
                            f"model_dict.lmbda={lmbda} "
                            f"model_dict.embed_size={embed_size} "
                            f"model_dict.use_annotator_features={use_annotator_features} "
                        )
        write_commands(
            file_name=file_name,
            commands=commands,
            model_name=model_name,
            data_type=data_type,
            data_set_name=data_set_name,
            mem=mem_dict[data_type],
            n_parallel_jobs=parallel_jobs_dict[data_type],
            cpus_per_task=cpus_per_task,
            slurm_logs_path=slurm_logs_path,
            slurm_error_logs_path=slurm_error_logs_path,
            use_slurm=use_slurm,
            use_gpu=accelerator_dict[data_set_name] == "gpu",
        )

# ================================================== CL parameters ====================================================
model_name = "cl"
print("\n\n" + model_name)
batch_path = f"./{model_name}"
if not os.path.exists(batch_path):
    os.makedirs(batch_path)

for data_set_name, data_type_list in data_type_dict.items():
    for data_type in data_type_list:
        if data_type == "inductive_25":
            continue
        commands = []
        file_name = f"{batch_path}/{model_name}_{data_set_name}_{data_type}.sh"
        for lr in lr_list:
            for weight_decay in weight_decay_list:
                for batch_size in batch_size_dict[data_set_name]:
                    commands.append(
                        f"{python_command} {run_experiment_path} with "
                        f"seed={seed} "
                        f"data_set_name={data_set_name} "
                        f"data_type={data_type} "
                        f"n_repeats={n_repeats} "
                        f"test_size={test_size} "
                        f"valid_size={valid_size} "
                        f"missing_label_ratio={missing_label_dict[data_set_name][data_type]} "
                        f"trainer_dict.max_epochs={max_epochs} "
                        f"trainer_dict.enable_progress_bar={enable_progress_bar} "
                        f"trainer_dict.enable_checkpointing={enable_checkpointing} "
                        f"trainer_dict.accelerator={accelerator_dict[data_set_name]} "
                        f"trainer_dict.devices={devices_dict[data_set_name]} "
                        f"trainer_dict.logger={logger} "
                        f"optimizer={optimizer} "
                        f"optimizer_dict.lr={lr} "
                        f"optimizer_dict.weight_decay={weight_decay} "
                        f"lr_scheduler={lr_scheduler} "
                        f"lr_scheduler_dict.T_max={max_epochs} "
                        f"batch_size={batch_size} "
                        f"dropout_rate={drop_out_rate_dict[data_set_name]} "
                        f"model_name={model_name} "
                    )
        write_commands(
            file_name=file_name,
            commands=commands,
            model_name=model_name,
            data_type=data_type,
            data_set_name=data_set_name,
            mem=mem_dict[data_type],
            n_parallel_jobs=parallel_jobs_dict[data_type],
            cpus_per_task=cpus_per_task,
            slurm_logs_path=slurm_logs_path,
            slurm_error_logs_path=slurm_error_logs_path,
            use_slurm=use_slurm,
            use_gpu=accelerator_dict[data_set_name] == "gpu",
        )


# ============================================== UNION-Net parameters =================================================
model_name = "union"
print("\n\n" + model_name)
batch_path = f"./{model_name}"
if not os.path.exists(batch_path):
    os.makedirs(batch_path)

epsilon = 0.00001

for data_set_name, data_type_list in data_type_dict.items():
    for data_type in data_type_list:
        if data_type == "inductive_25":
            continue
        commands = []
        file_name = f"{batch_path}/{model_name}_{data_set_name}_{data_type}.sh"
        for lr in lr_list:
            for weight_decay in weight_decay_list:
                for batch_size in batch_size_dict[data_set_name]:
                    commands.append(
                        f"{python_command} {run_experiment_path} with "
                        f"seed={seed} "
                        f"data_set_name={data_set_name} "
                        f"data_type={data_type} "
                        f"n_repeats={n_repeats} "
                        f"test_size={test_size} "
                        f"valid_size={valid_size} "
                        f"missing_label_ratio={missing_label_dict[data_set_name][data_type]} "
                        f"trainer_dict.max_epochs={max_epochs} "
                        f"trainer_dict.enable_progress_bar={enable_progress_bar} "
                        f"trainer_dict.enable_checkpointing={enable_checkpointing} "
                        f"trainer_dict.accelerator={accelerator_dict[data_set_name]} "
                        f"trainer_dict.devices={devices_dict[data_set_name]} "
                        f"trainer_dict.logger={logger} "
                        f"optimizer={optimizer} "
                        f"optimizer_dict.lr={lr} "
                        f"optimizer_dict.weight_decay={weight_decay} "
                        f"lr_scheduler={lr_scheduler} "
                        f"lr_scheduler_dict.T_max={max_epochs} "
                        f"batch_size={batch_size} "
                        f"dropout_rate={drop_out_rate_dict[data_set_name]} "
                        f"model_name={model_name} "
                        f"model_dict.epsilon={epsilon} "
                    )
        write_commands(
            file_name=file_name,
            commands=commands,
            model_name=model_name,
            data_type=data_type,
            data_set_name=data_set_name,
            mem=mem_dict[data_type],
            n_parallel_jobs=parallel_jobs_dict[data_type],
            cpus_per_task=cpus_per_task,
            slurm_logs_path=slurm_logs_path,
            slurm_error_logs_path=slurm_error_logs_path,
            use_slurm=use_slurm,
            use_gpu=accelerator_dict[data_set_name] == "gpu",
        )


# ============================================== REAC parameters =================================================
model_name = "reac"
print("\n\n" + model_name)
batch_path = f"./{model_name}"
if not os.path.exists(batch_path):
    os.makedirs(batch_path)

lmbda = 0.01

for data_set_name, data_type_list in data_type_dict.items():
    for data_type in data_type_list:
        if data_type == "inductive_25":
            continue
        commands = []
        file_name = f"{batch_path}/{model_name}_{data_set_name}_{data_type}.sh"
        for lr in lr_list:
            for weight_decay in weight_decay_list:
                for batch_size in batch_size_dict[data_set_name]:
                    commands.append(
                        f"{python_command} {run_experiment_path} with "
                        f"seed={seed} "
                        f"data_set_name={data_set_name} "
                        f"data_type={data_type} "
                        f"n_repeats={n_repeats} "
                        f"test_size={test_size} "
                        f"valid_size={valid_size} "
                        f"missing_label_ratio={missing_label_dict[data_set_name][data_type]} "
                        f"trainer_dict.max_epochs={max_epochs} "
                        f"trainer_dict.enable_progress_bar={enable_progress_bar} "
                        f"trainer_dict.enable_checkpointing={enable_checkpointing} "
                        f"trainer_dict.accelerator={accelerator_dict[data_set_name]} "
                        f"trainer_dict.devices={devices_dict[data_set_name]} "
                        f"trainer_dict.logger={logger} "
                        f"optimizer={optimizer} "
                        f"optimizer_dict.lr={lr} "
                        f"optimizer_dict.weight_decay={weight_decay} "
                        f"lr_scheduler={lr_scheduler} "
                        f"lr_scheduler_dict.T_max={max_epochs} "
                        f"batch_size={batch_size} "
                        f"dropout_rate={drop_out_rate_dict[data_set_name]} "
                        f"model_name={model_name} "
                        f"model_dict.lmbda={lmbda} "
                    )
        write_commands(
            file_name=file_name,
            commands=commands,
            model_name=model_name,
            data_type=data_type,
            data_set_name=data_set_name,
            mem=mem_dict[data_type],
            n_parallel_jobs=parallel_jobs_dict[data_type],
            cpus_per_task=cpus_per_task,
            slurm_logs_path=slurm_logs_path,
            slurm_error_logs_path=slurm_error_logs_path,
            use_slurm=use_slurm,
            use_gpu=accelerator_dict[data_set_name] == "gpu",
        )


# ================================================ LIA parameters =============================================
model_name = "lia"
max_epochs_lia = 200
ap_latent_dim = 16
n_em_steps = 7
n_fine_tune_epochs = 25
warm_start = True
lr_scheduler_lia = "CosineAnnealingRestarts"
T_0_lia = 25
T_mult_lia = 1

print("\n\n" + model_name)
batch_path = f"./{model_name}"
if not os.path.exists(batch_path):
    os.makedirs(batch_path)
for data_set_name, data_type_list in data_type_dict.items():
    for data_type in data_type_list:
        commands = []
        file_name = f"{batch_path}/{model_name}_{data_set_name}_{data_type}.sh"
        for use_annotator_features in use_annotator_features_dict[data_type]:
            for lr in lr_list:
                for weight_decay in weight_decay_list:
                    for batch_size in batch_size_dict[data_set_name]:
                        commands.append(
                            f"{python_command} {run_experiment_path} with "
                            f"seed={seed} "
                            f"data_set_name={data_set_name} "
                            f"data_type={data_type} "
                            f"n_repeats={n_repeats} "
                            f"test_size={test_size} "
                            f"valid_size={valid_size} "
                            f"missing_label_ratio={missing_label_dict[data_set_name][data_type]} "
                            f"trainer_dict.max_epochs={max_epochs_lia} "
                            f"trainer_dict.enable_progress_bar={enable_progress_bar} "
                            f"trainer_dict.enable_checkpointing={enable_checkpointing} "
                            f"trainer_dict.accelerator={accelerator_dict[data_set_name]} "
                            f"trainer_dict.devices={devices_dict[data_set_name]} "
                            f"trainer_dict.logger={logger} "
                            f"optimizer={optimizer} "
                            f"optimizer_dict.lr={lr} "
                            f"optimizer_dict.weight_decay={weight_decay} "
                            f"lr_scheduler={lr_scheduler_lia} "
                            f"lr_scheduler_dict.T_0={T_0_lia} "
                            f"lr_scheduler_dict.T_mult={T_mult_lia} "
                            f"batch_size={batch_size} "
                            f"dropout_rate={drop_out_rate_dict[data_set_name]} "
                            f"model_name={model_name} "
                            f"model_dict.ap_latent_dim={ap_latent_dim} "
                            f"model_dict.n_em_steps={n_em_steps} "
                            f"model_dict.n_fine_tune_epochs={n_fine_tune_epochs} "
                            f"model_dict.warm_start={warm_start} "
                            f"model_dict.use_annotator_features={use_annotator_features} "
                        )

        write_commands(
            file_name=file_name,
            commands=commands,
            model_name=model_name,
            data_type=data_type,
            data_set_name=data_set_name,
            mem=mem_dict[data_type],
            n_parallel_jobs=parallel_jobs_dict[data_type],
            cpus_per_task=cpus_per_task,
            slurm_logs_path=slurm_logs_path,
            slurm_error_logs_path=slurm_error_logs_path,
            use_slurm=use_slurm,
            use_gpu=accelerator_dict[data_set_name] == "gpu",
        )


# ================================================ Aggregation parameters =============================================
confusion_matrix = "full"
embed_size = 16
embed_x = "learned"
ap_use_residual = True
ap_use_outer_product = True

for model_name in ["gt", "mr"]:
    print("\n\n" + model_name)
    batch_path = f"./{model_name}"
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    for data_set_name, data_type_list in data_type_dict.items():
        for data_type in data_type_list:
            commands = []
            file_name = f"{batch_path}/{model_name}_{data_set_name}_{data_type}.sh"
            for use_annotator_features in [use_annotator_features_dict[data_type][0]]:
                for lr in lr_list:
                    for weight_decay in weight_decay_list:
                        for batch_size in batch_size_dict[data_set_name]:
                            commands.append(
                                f"{python_command} {run_experiment_path} with "
                                f"seed={seed} "
                                f"data_set_name={data_set_name} "
                                f"data_type={data_type} "
                                f"n_repeats={n_repeats} "
                                f"test_size={test_size} "
                                f"valid_size={valid_size} "
                                f"missing_label_ratio={missing_label_dict[data_set_name][data_type]} "
                                f"trainer_dict.max_epochs={max_epochs} "
                                f"trainer_dict.enable_progress_bar={enable_progress_bar} "
                                f"trainer_dict.enable_checkpointing={enable_checkpointing} "
                                f"trainer_dict.accelerator={accelerator_dict[data_set_name]} "
                                f"trainer_dict.devices={devices_dict[data_set_name]} "
                                f"trainer_dict.logger={logger} "
                                f"optimizer={optimizer} "
                                f"optimizer_dict.lr={lr} "
                                f"optimizer_dict.weight_decay={weight_decay} "
                                f"lr_scheduler={lr_scheduler} "
                                f"lr_scheduler_dict.T_max={max_epochs} "
                                f"batch_size={batch_size} "
                                f"dropout_rate={drop_out_rate_dict[data_set_name]} "
                                f"model_name={model_name} "
                                f"model_dict.confusion_matrix={confusion_matrix} "
                                f"model_dict.embed_size={embed_size} "
                                f"model_dict.embed_x={embed_x} "
                                f"model_dict.use_annotator_features={use_annotator_features} "
                                f"model_dict.ap_use_residual={ap_use_residual} "
                                f"model_dict.ap_use_outer_product={ap_use_outer_product}"
                            )

            write_commands(
                file_name=file_name,
                commands=commands,
                model_name=model_name,
                data_type=data_type,
                data_set_name=data_set_name,
                mem=mem_dict[data_type],
                n_parallel_jobs=parallel_jobs_dict[data_type],
                cpus_per_task=cpus_per_task,
                slurm_logs_path=slurm_logs_path,
                slurm_error_logs_path=slurm_error_logs_path,
                use_slurm=use_slurm,
                use_gpu=accelerator_dict[data_set_name] == "gpu",
            )
