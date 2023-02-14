from copy import deepcopy
from collections import OrderedDict

from evaluation.data_utils import (
    TABULAR_DATA_SETS,
    BW_IMAGE_DATA_SETS,
    RGB_IMAGE_DATA_SETS,
)

from lfma.classifiers import (
    MaDLClassifier,
    CrowdLayerClassifier,
    CoNALClassifier,
    AggregateClassifier,
    UnionNetClassifier,
    REACClassifier,
    LIAClassifier,
)
from lfma.modules import OuterProduct

from pytorch_lightning import seed_everything

from torch import nn
from torchvision.models import resnet18


# =============================================== MaDL Architecture ===================================================
def instantiate_madl_classifier(
    data_set_name,
    classes,
    n_features,
    n_ap_features,
    embed_x,
    ap_use_residual,
    ap_use_outer_product,
    eta,
    alpha,
    beta,
    embed_size,
    confusion_matrix,
    trainer_dict,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    dropout_rate,
    missing_label,
    random_state,
):
    # Set global seed for reproducibility.
    seed_everything(random_state, workers=True)

    # Number of classes.
    n_classes = len(classes)

    # Create ground truth net.
    gt_net_dict, n_hidden_neurons = get_gt_net(
        data_set_name=data_set_name,
        n_classes=n_classes,
        n_features=n_features,
        dropout_rate=dropout_rate,
    )

    # Create annotator performance modules.
    if confusion_matrix == "isotropic":
        ap_output_size = 1
    elif confusion_matrix == "diagonal":
        ap_output_size = n_classes
    elif confusion_matrix == "full":
        ap_output_size = n_classes * n_classes
    else:
        raise ValueError("'confusion_matrix' must be in ['isotropic', 'diagonal', 'full'].")

    ap_embed_x = None
    use_gt_embed_x = False
    n_embed_features = embed_size
    ap_outer_product = None

    ap_embed_a = nn.Sequential(
        nn.Linear(in_features=n_ap_features, out_features=embed_size),
    )
    if embed_x == "none":
        ap_hidden = nn.Identity()

    elif embed_x in ["raw", "learned"]:
        n_embed_features += embed_size
        if embed_x == "raw":
            ap_embed_x = nn.Sequential(
                deepcopy(gt_net_dict["gt_embed_x"]),
                nn.Linear(in_features=n_hidden_neurons, out_features=embed_size),
            )
        else:
            use_gt_embed_x = True
            ap_embed_x = nn.Linear(in_features=n_hidden_neurons, out_features=embed_size)

        if ap_use_outer_product:
            ap_outer_product = nn.Sequential(
                OuterProduct(embedding_size=embed_size, output_size=embed_size),
            )
            n_embed_features += embed_size

        ap_hidden = nn.Sequential(
            nn.Linear(in_features=n_embed_features, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=embed_size),
            nn.BatchNorm1d(embed_size),
        )
    else:
        raise ValueError("'embed_x' must be in ['none', 'raw', 'learned'].")
    ap_output = nn.Linear(in_features=embed_size, out_features=ap_output_size)

    module_dict = {
        "gt_embed_x": gt_net_dict["gt_embed_x"],
        "gt_mlp": gt_net_dict["gt_mlp"],
        "ap_embed_a": ap_embed_a,
        "ap_outer_product": ap_outer_product,
        "ap_use_residual": ap_use_residual,
        "ap_hidden": ap_hidden,
        "ap_output": ap_output,
        "ap_embed_x": ap_embed_x,
        "ap_use_gt_embed_x": use_gt_embed_x,
        "eta": eta,
        "alpha": alpha,
        "beta": beta,
        "confusion_matrix": confusion_matrix,
        "optimizer": optimizer,
        "optimizer_dict": optimizer_dict,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_dict": lr_scheduler_dict,
    }

    madl = MaDLClassifier(
        module_dict=module_dict,
        trainer_dict=trainer_dict,
        classes=classes,
        missing_label=missing_label,
        random_state=random_state,
    )
    return madl


# =========================================== CrowdLayer Architecture =================================================
def instantiate_crowd_layer_classifier(
    data_set_name,
    classes,
    n_features,
    trainer_dict,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    missing_label,
    dropout_rate,
    random_state,
):
    # Set global seed for reproducibility.
    seed_everything(random_state, workers=True)

    # Create ground truth net.
    gt_net_dict, n_hidden_neurons = get_gt_net(
        data_set_name=data_set_name,
        n_classes=len(classes),
        n_features=n_features,
        dropout_rate=dropout_rate,
    )

    module_dict = {
        "gt_net": nn.Sequential(gt_net_dict),
        "optimizer": optimizer,
        "optimizer_dict": optimizer_dict,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_dict": lr_scheduler_dict,
    }

    crowd_layer = CrowdLayerClassifier(
        module_dict=module_dict,
        trainer_dict=trainer_dict,
        classes=classes,
        missing_label=missing_label,
        random_state=random_state,
    )

    return crowd_layer


# ============================================= REAC Architecture =================================================
def instantiate_reac_classifier(
    data_set_name,
    classes,
    n_features,
    trainer_dict,
    lmbda,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    missing_label,
    dropout_rate,
    random_state,
):
    # Set global seed for reproducibility.
    seed_everything(random_state, workers=True)

    # Create ground truth net.
    gt_net_dict, n_hidden_neurons = get_gt_net(
        data_set_name=data_set_name,
        n_classes=len(classes),
        n_features=n_features,
        dropout_rate=dropout_rate,
    )

    module_dict = {
        "gt_net": nn.Sequential(gt_net_dict),
        "lmbda": lmbda,
        "optimizer": optimizer,
        "optimizer_dict": optimizer_dict,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_dict": lr_scheduler_dict,
    }

    reac = REACClassifier(
        module_dict=module_dict,
        trainer_dict=trainer_dict,
        classes=classes,
        missing_label=missing_label,
        random_state=random_state,
    )

    return reac


# ============================================= UnionNet Architecture =================================================
def instantiate_union_net_classifier(
    data_set_name,
    classes,
    n_features,
    trainer_dict,
    epsilon,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    missing_label,
    dropout_rate,
    random_state,
):
    # Set global seed for reproducibility.
    seed_everything(random_state, workers=True)

    # Create ground truth net.
    gt_net_dict, n_hidden_neurons = get_gt_net(
        data_set_name=data_set_name,
        n_classes=len(classes),
        n_features=n_features,
        dropout_rate=dropout_rate,
    )

    module_dict = {
        "gt_net": nn.Sequential(gt_net_dict),
        "epsilon": epsilon,
        "optimizer": optimizer,
        "optimizer_dict": optimizer_dict,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_dict": lr_scheduler_dict,
    }

    union_net = UnionNetClassifier(
        module_dict=module_dict,
        trainer_dict=trainer_dict,
        classes=classes,
        missing_label=missing_label,
        random_state=random_state,
    )

    return union_net


# =============================================== CoNAL Architecture ==================================================
def instantiate_conal_classifier(
    data_set_name,
    classes,
    n_features,
    n_ap_features,
    lmbda,
    embed_size,
    trainer_dict,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    dropout_rate,
    missing_label,
    random_state,
):
    # Set global seed for reproducibility.
    seed_everything(random_state, workers=True)

    # Number of classes.
    n_classes = len(classes)

    # Create ground truth net.
    gt_net_dict, n_hidden_neurons = get_gt_net(
        data_set_name=data_set_name,
        n_classes=n_classes,
        n_features=n_features,
        dropout_rate=dropout_rate,
    )

    ap_embed_a = nn.Sequential(
        nn.Linear(in_features=n_ap_features, out_features=embed_size),
    )
    ap_embed_x = nn.Sequential(
        deepcopy(gt_net_dict["gt_embed_x"][:-1]),
        nn.Linear(in_features=n_hidden_neurons, out_features=embed_size),
    )

    module_dict = {
        "gt_net": nn.Sequential(gt_net_dict),
        "ap_embed_a": ap_embed_a,
        "ap_embed_x": ap_embed_x,
        "lmbda": lmbda,
        "optimizer": optimizer,
        "optimizer_dict": optimizer_dict,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_dict": lr_scheduler_dict,
    }

    conal = CoNALClassifier(
        module_dict=module_dict,
        trainer_dict=trainer_dict,
        classes=classes,
        missing_label=missing_label,
        random_state=random_state,
    )
    return conal


# ================================================== LIA Architecture =================================================
def instantiate_lia_classifier(
    data_set_name,
    classes,
    n_features,
    n_ap_features,
    ap_latent_dim,
    n_em_steps,
    warm_start,
    n_fine_tune_epochs,
    trainer_dict,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    dropout_rate,
    missing_label,
    random_state,
):
    # Set global seed for reproducibility.
    seed_everything(random_state, workers=True)

    # Number of classes.
    n_classes = len(classes)

    # Create ground truth net.
    gt_net_dict, n_hidden_neurons = get_gt_net(
        data_set_name=data_set_name,
        n_classes=n_classes,
        n_features=n_features,
        dropout_rate=dropout_rate,
    )

    ap_difficulty_layer = nn.Sequential(
        nn.Linear(in_features=n_hidden_neurons, out_features=128),
        nn.BatchNorm1d(num_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=ap_latent_dim * n_classes ** 2),
    )
    ap_competence_layer = nn.Linear(in_features=n_ap_features, out_features=ap_latent_dim * n_classes ** 2)

    module_dict = {
        "gt_embed_x": gt_net_dict["gt_embed_x"],
        "gt_mlp": gt_net_dict["gt_mlp"],
        "ap_difficulty_layer": ap_difficulty_layer,
        "ap_competence_layer": ap_competence_layer,
        "ap_latent_dim": ap_latent_dim,
        "n_em_steps": n_em_steps,
        "warm_start": warm_start,
        "n_fine_tune_epochs": n_fine_tune_epochs,
        "optimizer": optimizer,
        "optimizer_dict": optimizer_dict,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_dict": lr_scheduler_dict,
    }

    lia = LIAClassifier(
        module_dict=module_dict,
        trainer_dict=trainer_dict,
        classes=classes,
        missing_label=missing_label,
        random_state=random_state,
    )
    return lia


# ============================================= Aggregate Architecture ================================================
def instantiate_aggregate_classifier(
    data_set_name,
    classes,
    n_features,
    n_ap_features,
    embed_x,
    ap_use_residual,
    ap_use_outer_product,
    embed_size,
    confusion_matrix,
    trainer_dict,
    optimizer,
    optimizer_dict,
    lr_scheduler,
    lr_scheduler_dict,
    dropout_rate,
    missing_label,
    random_state,
):
    # Set global seed for reproducibility.
    seed_everything(random_state, workers=True)

    # Number of classes.
    n_classes = len(classes)

    # Create ground truth net.
    gt_net_dict, n_hidden_neurons = get_gt_net(
        data_set_name=data_set_name,
        n_classes=n_classes,
        n_features=n_features,
        dropout_rate=dropout_rate,
    )

    # Create annotator performance modules.
    if confusion_matrix == "isotropic":
        ap_output_size = 1
    elif confusion_matrix == "diagonal":
        ap_output_size = n_classes
    elif confusion_matrix == "full":
        ap_output_size = n_classes * n_classes
    else:
        raise ValueError("'confusion_matrix' must be in ['isotropic', 'diagonal', 'full'].")

    ap_embed_x = None
    use_gt_embed_x = False
    n_embed_features = embed_size
    ap_outer_product = None

    ap_embed_a = nn.Linear(in_features=n_ap_features, out_features=embed_size)
    if embed_x == "none":
        ap_hidden = nn.Identity()

    elif embed_x in ["raw", "learned"]:
        n_embed_features += embed_size
        if embed_x == "raw":
            ap_embed_x = nn.Sequential(
                deepcopy(gt_net_dict["gt_embed_x"]),
                nn.Linear(in_features=n_hidden_neurons, out_features=embed_size),
            )
        else:
            use_gt_embed_x = True
            ap_embed_x = nn.Linear(in_features=n_hidden_neurons, out_features=embed_size)

        if ap_use_outer_product:
            ap_outer_product = OuterProduct(embedding_size=embed_size, output_size=embed_size)
            n_embed_features += embed_size

        ap_hidden = nn.Sequential(
            nn.Linear(in_features=n_embed_features, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=embed_size),
            nn.BatchNorm1d(embed_size),
        )
    else:
        raise ValueError("'embed_x' must be in ['none', 'raw', 'learned'].")
    ap_output = nn.Linear(in_features=embed_size, out_features=ap_output_size)

    module_dict = {
        "gt_embed_x": gt_net_dict["gt_embed_x"],
        "gt_mlp": gt_net_dict["gt_mlp"],
        "ap_embed_a": ap_embed_a,
        "ap_outer_product": ap_outer_product,
        "ap_hidden": ap_hidden,
        "ap_output": ap_output,
        "ap_embed_x": ap_embed_x,
        "ap_use_gt_embed_x": use_gt_embed_x,
        "ap_use_residual": ap_use_residual,
        "confusion_matrix": confusion_matrix,
        "optimizer": optimizer,
        "optimizer_dict": optimizer_dict,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_dict": lr_scheduler_dict,
    }

    agg_clf = AggregateClassifier(
        module_dict=module_dict,
        trainer_dict=trainer_dict,
        classes=classes,
        missing_label=missing_label,
        random_state=random_state,
    )
    return agg_clf


# =============================================== GT Architecture =====================================================
def get_gt_net(data_set_name, n_classes, n_features, dropout_rate, pretrained=False):
    gt_net_ordered_dict = OrderedDict()
    if data_set_name in TABULAR_DATA_SETS:
        # Create ground truth modules.
        n_hidden_neurons = 128
        gt_net_ordered_dict["gt_embed_x"] = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_hidden_neurons),
            nn.BatchNorm1d(num_features=n_hidden_neurons),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    elif data_set_name in BW_IMAGE_DATA_SETS:
        n_hidden_neurons = 84
        gt_net_ordered_dict["gt_embed_x"] = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    elif data_set_name in RGB_IMAGE_DATA_SETS:
        n_hidden_neurons = 512
        resnet = resnet18(pretrained=pretrained)
        # Init layer does not have a kernel size of 7 since cifar has a smaller
        # size of 32x32
        if not pretrained:
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            resnet.maxpool = nn.Identity()
        children_list = []
        for n, c in resnet.named_children():
            children_list.append(c)
            if n == "avgpool":
                break
        children_list.append(nn.Flatten())
        children_list.append(nn.Dropout(p=dropout_rate))
        gt_net_ordered_dict["gt_embed_x"] = nn.Sequential(*children_list)
    else:
        raise ValueError(
            f"{data_set_name} must be in " f"{TABULAR_DATA_SETS + BW_IMAGE_DATA_SETS + RGB_IMAGE_DATA_SETS}."
        )

    gt_net_ordered_dict["gt_mlp"] = nn.Linear(in_features=n_hidden_neurons, out_features=n_classes)
    return gt_net_ordered_dict, n_hidden_neurons
