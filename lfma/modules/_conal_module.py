import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from pytorch_lightning import LightningModule


class CoNALModule(LightningModule):
    """CoNALModule

    CoNAL (Common Noise Adaption Layers) [1] is an end-to-end learning solution with two types of noise adaptation
    layers: one is shared across annotators to capture their commonly shared confusions, and the other one is
    pertaining to each annotator to realize individual confusion.

    Parameters
    ----------
    n_classes : int
        Number of classes
    gt_net : nn.Module
        Pytorch module of the GT model embedding the input samples.
    ap_embed_a : nn.Module
        Pytorch module of the AP model embedding the annotator features for the AP model.
    ap_embed_x : nn.Module, optional (default=None)
        Pytorch module of the AP model embedding samples.
    A : torch.tensor of shape (n_annotators, n_annotator_features)
        The matrix `A` is the annotator feature matrix. The exact form depends on the `module_class`.
        If it is `None`, a one-hot encoding is used to differentiate between annotators.
    lmbda : float, optional (default=1e-5)
        Regularization parameter to enforce the common and individual confusion matrices of the annotators to be
        different.
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
    [1] Chu, Z., Ma, J., & Wang, H. (2021, May). Learning from Crowds by Modeling Common Confusions.
        In AAAI (pp. 5832-5840).
    """

    def __init__(
        self,
        n_classes,
        n_annotators,
        gt_net,
        ap_embed_a,
        ap_embed_x,
        A=None,
        lmbda=1e-5,
        optimizer=None,
        optimizer_dict=None,
        lr_scheduler=None,
        lr_scheduler_dict=None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.gt_net = gt_net
        self.ap_embed_a = ap_embed_a
        self.ap_embed_x = ap_embed_x
        self.register_buffer("A", torch.eye(n_annotators) if A is None else A)
        self.lmbda = lmbda
        self.kernel = nn.Parameter(torch.stack([2 * torch.eye(n_classes)] * n_annotators))
        self.common_kernel = nn.Parameter(2.0 * torch.eye(n_classes).float())
        self.optimizer = optimizer
        self.optimizer_dict = optimizer_dict
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_dict = lr_scheduler_dict
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
        # Compute logits.
        logit_class = self.gt_net(x)

        # Compute class-membership probabilities.
        p_class = F.softmax(logit_class, dim=-1)

        if not return_logits_annot:
            return p_class

        # Compute embeddings and normalize them.
        x_embedding = F.normalize(self.ap_embed_x(x))
        a_embedding = F.normalize(self.ap_embed_a(self.A))

        # Take product of embeddings to compute probability of common kernel.
        common_rate = torch.einsum("ij,kj->ik", (x_embedding, a_embedding))
        common_rate = F.sigmoid(common_rate)

        # Compute common kernel and individual kernel products.
        logits_common = torch.einsum("ij,jk->ik", (p_class, self.common_kernel))
        logits_individual = torch.einsum("ik,jkl->ijl", (p_class, self.kernel))

        # Compute logits per annotator.
        logits_annot = common_rate[:, :, None] * logits_common[:, None, :]
        logits_annot += (1 - common_rate[:, :, None]) * logits_individual
        logits_annot = logits_annot.transpose(1, 2)

        return p_class, logits_annot

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        _, logits_annot = self.forward(x)
        loss = F.cross_entropy(logits_annot, y, reduction="mean", ignore_index=-1)
        if self.lmbda > 0:
            diff = (self.kernel - self.common_kernel).view(y.shape[1], -1)
            norm_sum = diff.norm(dim=1, p=2).sum()
            loss -= self.lmbda * norm_sum
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits_class = self.forward(x, return_logits_annot=False)
        y_pred = logits_class.argmax(dim=-1)
        acc = (y_pred == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def predict_step(self, batch, batch_idx):
        p_class, logits_annot = self.forward(batch)
        p_annot = F.softmax(logits_annot, dim=1)
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
