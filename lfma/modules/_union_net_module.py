import torch

from pytorch_lightning import LightningModule

from torch.optim import AdamW
from torch import nn
from torch.nn import functional as F


class UnionNetModule(LightningModule):
    """UnionNetModule

    UnionNet [1] concatenates the one-hot encoded vectors of labels provided by all the annotators, which takes all the
    labeling information as a union and coordinates multiple annotators.

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
    [1] Wei, Hongxin, Renchunzi Xie, Lei Feng, Bo Han, and Bo An. "Deep Learning From Multiple Noisy Annotators as A
        Union." IEEE Transactions on Neural Networks and Learning Systems (2022).
    """

    def __init__(
        self,
        n_classes,
        n_annotators,
        gt_net,
        epsilon=1e-5,
        optimizer=None,
        optimizer_dict=None,
        lr_scheduler=None,
        lr_scheduler_dict=None,
    ):
        super(UnionNetModule, self).__init__()
        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.gt_net = gt_net
        self.optimizer = optimizer
        self.optimizer_dict = optimizer_dict
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_dict = lr_scheduler_dict

        # Create transition matrix of the UnionNet.
        init_matrix = (1 - epsilon) * torch.eye(n_classes) + epsilon / (n_classes - 1) * (1 - torch.eye(n_classes))
        self.transition_matrix = nn.Parameter(torch.concat([init_matrix for _ in range(n_annotators)]))

        self.save_hyperparameters()

    def forward(self, x, return_p_annot=True):
        """Forward propagation of samples through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Samples.
        return_p_annot: bool, optional (default=True)
            Flag whether the annotation probabilities are to be returned, next to the class-membership probabilities.

        Returns
        -------
        p_class : torch.Tensor of shape (batch_size, n_classes)
            Class-membership probabilities.
        p_annot : torch.Tensor of shape (batch_size, n_classes * n_annotators)
            Annotation probabilities for the annotators as a union.
        """
        # Compute logits.
        logit_class = self.gt_net(x)

        # Compute class-membership probabilities.
        p_class = F.softmax(logit_class, dim=-1)

        if not return_p_annot:
            return p_class

        # Compute logits per annotator.
        p_annot = p_class @ F.softmax(self.transition_matrix, dim=0).T

        return p_class, p_annot

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_one_hot = F.one_hot(y + 1, num_classes=self.n_classes + 1)[:, :, 1:].float()
        y_one_hot = y_one_hot.flatten(start_dim=1, end_dim=2)
        _, p_annot = self.forward(x)
        loss = (-y_one_hot * p_annot.log()).mean()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p_class = self.forward(x, return_p_annot=False)
        y_pred = p_class.argmax(dim=-1)
        acc = (y_pred == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def predict_step(self, batch, batch_idx):
        p_class, p_annot = self.forward(batch)
        p_annot = p_annot.reshape(-1, self.n_annotators, self.n_classes)
        p_annot /= p_annot.sum(dim=-1, keepdims=True)
        p_annot = p_annot.swapaxes(1, 2)
        return p_class, p_annot

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
