import math
import torch

from torch.optim import AdamW
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule


class AggregateModule(LightningModule):
    """AggregateModule

    This class implements a model (based on the MaDL [1] architecture), which separately trains a ground truth (GT)
    model for classification and an annotator performance (AP) model for given aggregated class labels.

    Parameters
    ----------
    n_classes : int
        Number of classes
    gt_embed_x : nn.Module
        Pytorch module of the GT model embedding the input samples.
    gt_mlp : nn.Module
        Pytorch module of the GT model taking the embedding the samples as input to predict class-membership logits.
    ap_embed_a : nn.Module
        Pytorch module of the AP model embedding the annotator features for the AP model.
    ap_output : nn.Module
        Pytorch module of the AP model predicting the logits of the conditional confusion matrix
    ap_embed_x : nn.Module, optional (default=None)
        Pytorch module of the AP model embedding samples.
    ap_outer_product : nn.Module, optional (default=None)
        Outer product-based layer to model interactions between annotator and sample embeddings. By default, it is None
        and therefore not used.
    ap_hidden : nn.Module, optional (default=None)
        Pytorch module of the AP model taking the concatenation of annotator, sample (optional) and outer product
        (optional) embedding as input to create a new embedding as input to the `ap_output` module.
        By default, it is an identity mapping.
    ap_use_gt_embed_x : bool, optional (default=True)
        Flag whether the learned sample embeddings or the raw samples are used as inputs to `ap_embed_x`. By default,
        it is True and only relevant, if `ap_embed_x` is not None.
    ap_use_residual : bool, optional (default=True)
        Flag whether a residual block is to be applied to the output of the `ap_hidden`. By default, it is True
        and only relevant, if `ap_hidden` is not None.
    confusion_matrix : {'isotropic', 'diagonal', 'full'}
        Determines the type of estimated confusion matrix, where we differ between:
            - 'isotropic' corresponding to a scalar (one output neuron for `ap_output`) from which the confusion matrix
              is constructed,
            - 'diagonal' corresponding to a vector as diagonal (`n_classes` output neurons for `ap_output`) from which
              the confusion matrix is constructed,
            - 'full' corresponding to a vector (`n_classes * n_classes` output neurons for `ap_output`) from which
              the confusion matrix is constructed.
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
    [1] Herde, Marek, Huseljic, Denis, and Sick, Bernhard. "Mulit-annotator Deep Learning: A Modular Probabilistic
        Framework."
    """

    def __init__(
        self,
        n_classes,
        gt_embed_x,
        gt_mlp,
        ap_embed_a,
        ap_output,
        ap_outer_product=None,
        ap_hidden=None,
        ap_embed_x=None,
        ap_use_gt_embed_x=False,
        ap_use_residual=False,
        confusion_matrix="full",
        optimizer=None,
        optimizer_dict=None,
        lr_scheduler=None,
        lr_scheduler_dict=None,
    ):
        super(AggregateModule, self).__init__()
        self.n_classes = n_classes
        self.gt_embed_x = gt_embed_x
        self.gt_mlp = gt_mlp
        self.ap_embed_a = ap_embed_a
        self.ap_output = ap_output
        self.ap_hidden = nn.Identity() if ap_hidden is None else ap_hidden
        self.ap_embed_x = ap_embed_x
        self.ap_use_gt_embed_x = ap_use_gt_embed_x
        self.ap_use_residual = ap_use_residual
        self.ap_outer_product = ap_outer_product
        self.optimizer = optimizer
        self.optimizer_dict = optimizer_dict
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_dict = lr_scheduler_dict
        self.confusion_matrix = confusion_matrix
        self.register_buffer("eye", torch.eye(self.n_classes))
        self.register_buffer("one_minus_eye", 1 - self.eye)

        # Save hyper parameters.
        self.save_hyperparameters()

    def forward(self, x, a=None):
        """Forward propagation of samples' and annotators' (optional) features through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Sample features.
        a : torch.Tensor of shape (n_annotators, *), optional (default=None)
            Annotator features, which are None by default. In this case, only the samples are forward propagated
            through the GT model.

        Returns
        -------
        logits_class : torch.Tensor of shape (batch_size, n_classes)
            Class-membership logits.
        logits_perf : torch.Tensor of shape (batch_size, n_annotators, n_classes, n_classes)
            Logits of conditional confusion matrices as proxies of the annotators' performances.
        """
        # Compute feature embedding for classifier.
        x_learned = self.gt_embed_x(x)

        # Compute class-membership probabilities.
        logits_class = self.gt_mlp(x_learned)

        if a is None:
            return logits_class

        # Compute annotator performances per annotator.
        a = a[0]
        n_annotators = a.shape[0]
        n_samples = x.shape[0]
        combs = torch.cartesian_prod(torch.arange(n_samples), torch.arange(n_annotators))

        # Compute annotator embedding.
        annot_embeddings = self.ap_embed_a(a)[combs[:, 1]]
        embeddings = [annot_embeddings]

        # Optionally: Compute feature embeddings and combine with annotator
        # embeddings.
        if self.ap_embed_x is not None:
            if self.ap_use_gt_embed_x:
                embeddings.append(self.ap_embed_x(x_learned.detach())[combs[:, 0]])
            else:
                embeddings.append(self.ap_embed_x(x.detach())[combs[:, 0]])
            if self.ap_outer_product is not None:
                product = self.ap_outer_product(torch.stack(embeddings, dim=1))
                embeddings.append(product)
        concat_embeddings = torch.concat(embeddings, dim=-1)

        # Propagate embedding through hidden layers.
        embeddings = self.ap_hidden(concat_embeddings)

        # Optionally: Add annotator embeddings as residuals.
        if self.ap_use_residual:
            embeddings = F.relu(embeddings + annot_embeddings)

        # Compute logits of annotator performances.
        logits_perf = self.ap_output(embeddings)
        logits_perf = logits_perf.reshape((n_samples, n_annotators, -1))

        # Transform logits of annotator performances into confusion matrix.
        if self.confusion_matrix in ["isotropic", "diagonal"]:
            logits_perf = self.eye[None, None] * logits_perf[:, :, :, None] + self.one_minus_eye[None, None] * (
                -math.log(self.n_classes - 1)
            )
        elif self.confusion_matrix == "full":
            logits_perf = logits_perf.view(-1, n_annotators, self.n_classes, self.n_classes)
        return logits_class, logits_perf

    def training_step(self, train_batch, batch_idx):
        x, y, a = train_batch
        logits_class, logits_perf = self.forward(x, a)
        loss = self._annot_cross_entropy_loss(y=y, logits_class=logits_class, logits_perf=logits_perf)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits_class = self.forward(x, None)
        y_pred = logits_class.argmax(dim=-1)
        acc = (y_pred == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list):
            x, a = batch
            logit_class, logit_perf = self.forward(x, a)
            p_class = F.softmax(logit_class, dim=-1)
            p_perf = F.softmax(logit_perf, dim=-1)
            return p_class, p_perf
        x = batch
        logit_class = self.forward(x, None)
        p_class = F.softmax(logit_class, dim=-1)
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

    def _annot_cross_entropy_loss(self, y, logits_class, logits_perf):
        # Compute prediction loss for GT model.
        y_gt = y[:, -1]
        p_class_log = F.log_softmax(logits_class, dim=-1)
        loss_gt = F.nll_loss(p_class_log, y_gt, reduction="sum", ignore_index=-1)
        loss_gt = loss_gt / (y_gt != -1).sum().float().item()

        # Compute prediction loss for AP model.
        y_ap = y[:, :-1]
        p_perf_log = F.log_softmax(logits_perf, dim=-1)
        p_perf_log = p_perf_log[torch.arange(len(y_ap)), :, y_gt, :].swapaxes(1, 2)
        loss_ap = F.nll_loss(p_perf_log, y_ap, reduction="sum", ignore_index=-1)
        loss_ap = loss_ap / (y_ap != -1).sum().float().item()
        return loss_gt + loss_ap
