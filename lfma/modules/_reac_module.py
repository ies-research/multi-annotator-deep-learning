import torch

from pytorch_lightning import LightningModule

from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW


class REACModule(LightningModule):
    """REACModule

    REAC (Regularized Estimation of Annotator Confusion) [1] jointly learns the individual confusion matrix of each
    annotator and the underlying GT distribution. Therefor, a regularization term is added to the loss function that
    encourages convergence to the true annotator confusion matrix.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    n_annotators : int
        Number of annotators.
    gt_net : nn.Module
        Pytorch module of the GT model taking samples as input to predict class-membership logits.
    lmbda : non-negative float, optional (default=0.01)
        Regularization term penalizing the sums of diagonals of annotators' confusion matrices.
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
    [1] Tanno, Ryutaro, Ardavan Saeedi, Swami Sankaranarayanan, Daniel C. Alexander, and Nathan Silberman.
        "Learning from noisy labels by regularized estimation of annotator confusion." In Proceedings of the IEEE/CVF
        conference on computer vision and pattern recognition, pp. 11244-11253. 2019.
    """

    def __init__(
        self,
        n_classes,
        n_annotators,
        gt_net,
        lmbda=0.01,
        optimizer=None,
        optimizer_dict=None,
        lr_scheduler=None,
        lr_scheduler_dict=None,
    ):
        super(REACModule, self).__init__()
        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.gt_net = gt_net
        self.lmbda = lmbda
        self.confusion_matrices = nn.Parameter(torch.stack([6.0 * torch.eye(n_classes) - 5.0] * n_annotators))
        self.optimizer = optimizer
        self.optimizer_dict = optimizer_dict
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_dict = lr_scheduler_dict
        self.save_hyperparameters()

    def forward(self, x):
        """Forward propagation of samples through the GT model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Samples.

        Returns
        -------
        logits_class : torch.Tensor of shape (batch_size, n_classes)
            Class-membership logits.
        """
        # Compute logits.
        logits_class = self.gt_net(x)

        return logits_class

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits_class = self.forward(x)
        p_class_log = F.log_softmax(logits_class, dim=-1)
        p_perf_log = torch.log_softmax(self.confusion_matrices, dim=-1)
        p_annot_log = torch.logsumexp(p_class_log[:, None, :, None] + p_perf_log[None, :, :, :], dim=2).swapaxes(1, 2)
        loss = F.nll_loss(p_annot_log, y, reduction="mean", ignore_index=-1)
        if self.lmbda > 0:
            p_perf = F.softmax(self.confusion_matrices, dim=-1)
            trace = p_perf.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1).mean()
            loss += self.lmbda * trace
        loss /= len(x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p_class = self.forward(x)
        y_pred = p_class.argmax(dim=-1)
        acc = (y_pred == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def predict_step(self, batch, batch_idx):
        p_class = F.softmax(self.forward(batch), dim=-1)
        return p_class

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
