import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from torch.optim import AdamW


class CrowdLayerModule(LightningModule):
    """CrowdLayerModule

    CrowdLayer [1] is a layer added at the end of a classifying neural network and allows us to train deep neural
    networks end-to-end, directly from the noisy labels of multiple annotators, using only backpropagation.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    n_annotators : int
        Number of annotators.
    gt_net : nn.Module
        Pytorch module of the GT model taking samples as input to predict class-membership logits.
    epsilon : non-negative float, optional (default=1e-5)
        Prior error probability to initialize annotators' confusion matrices.
    optimizer : torch.optim.Optimizer, optional (default=None)
        Optimizer responsible for optimizing the GT and AP parameters. If None, the `AdamW` optimizer is used by
        default.
    optimizer_dict : dict, optional (default=None)
        Parameters passed to `optimizer`.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler, optional (default=None)
        Optimizer responsible for optimizing the GT and AP parameters. If None, the `AdamW` optimizer is used by
        default.
    lr_scheduler_dict : dict, optional (default=None)
        Parameters passed to `lr_scheduler`.

    References
    ----------
    [1] Rodrigues, Filipe, and Francisco Pereira. "Deep learning from crowds." In Proceedings of the AAAI conference on
        artificial intelligence, vol. 32, no. 1. 2018.
    """

    def __init__(
        self,
        n_classes,
        n_annotators,
        gt_net,
        optimizer=None,
        optimizer_dict=None,
        lr_scheduler=None,
        lr_scheduler_dict=None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.gt_net = gt_net
        self.optimizer = optimizer
        self.optimizer_dict = optimizer_dict
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_dict = lr_scheduler_dict

        # Setup crowd layer.
        self.annotator_layers = nn.ModuleList()
        for i in range(n_annotators):
            layer = nn.Linear(n_classes, n_classes, bias=False)
            nn.init.eye_(layer.weight)
            self.annotator_layers.append(layer)

        self.save_hyperparameters()

    def forward(self, x, return_logits_annot=True):
        """Forward propagation of samples through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Samples.
        return_logits_annot: bool, optional (default=True)
            Flag whether the annotation logits are to be returned, next to the class-membership probabilities.

        Returns
        -------
        p_class : torch.Tensor of shape (batch_size, n_classes)
            Class-membership probabilities.
        logits_annot : torch.Tensor of shape (batch_size, n_annotators, n_classes)
            Annotation logits for each sample-annotator pair.
        """
        # Compute class-membership logits.
        logit_class = self.gt_net(x)

        # Compute class-membership probabilities.
        p_class = F.softmax(logit_class, dim=-1)

        if not return_logits_annot:
            return p_class

        # Compute logits per annotator.
        logits_annot = []
        for layer in self.annotator_layers:
            logits_annot.append(layer(p_class))
        logits_annot = torch.stack(logits_annot, dim=2)

        return p_class, logits_annot

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        _, logits_annot = self.forward(x)
        loss = F.cross_entropy(logits_annot, y, reduction="mean", ignore_index=-1)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p_class = self.forward(x, return_logits_annot=False)
        y_pred = p_class.argmax(dim=-1)
        acc = (y_pred == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def predict_step(self, batch, batch_idx):
        P_class, P_annot = self.forward(batch)
        P_annot = F.softmax(P_annot, dim=1)
        return P_class, P_annot

    def configure_optimizers(self):
        # Setup optimizer.
        optimizer = AdamW if self.optimizer is None else self.optimizer
        optimizer_dict = {} if self.optimizer_dict is None else self.optimizer_dict
        optimizer = optimizer(self.parameters(), **optimizer_dict)

        # Return optimizer, if no learning rate scheduler has been defined.
        if self.lr_scheduler is None:
            return [optimizer]

        lr_scheduler_dict = {} if self.lr_scheduler_dict is None else self.lr_scheduler_dict
        lr_scheduler = self.lr_scheduler(optimizer, **lr_scheduler_dict)
        return [optimizer], [lr_scheduler]
