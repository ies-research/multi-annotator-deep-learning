import torch

from pytorch_lightning import LightningModule

from torch.optim import AdamW
from torch import nn
from torch.nn import functional as F


class LIAModule(LightningModule):
    """LIAModule

    This class implements the framework "learning from imperfect annotations" (LIA) [1], which trains a ground
    truth (GT) model for classification and an annotator performance (AP) model via the EM-algorithm.

    Parameters
    ----------
    n_classes : int
        Number of classes
    gt_embed_x : nn.Module
        Pytorch module of the GT model embedding the input samples.
    gt_mlp : nn.Module
        Pytorch module of the GT model taking the embedding the samples as input to predict class-membership logits.
    ap_difficulty_layer : nn.Module, optional (default=None)
        Pytorch module of the AP model learning latent representations of samples' difficulties regarding annotating.
    ap_competence_layer : nn.Module
        Pytorch module of the AP model learning latent representations of annotators' competences.
    ap_latent_dim : int
        Latent dimension of the representations learned by the `ap_difficulty_layer` and `ap_competence_layer`.
    n_em_steps : int, optional (default=True)
        Number of EM-steps.
    warm_start : bool, optional (default=True)
        Flag whether the parameters of the GT and AP model are to be used as initialization for the next EM-step or
        whether they are to be re-initialized after each EM-step.
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
    [1] Platanios, Emmanouil Antonios, Maruan Al-Shedivat, Eric Xing, and Tom Mitchell.
        "Learning from imperfect annotations." arXiv preprint arXiv:2004.03473 (2020).
    """

    def __init__(
        self,
        n_classes,
        gt_embed_x,
        gt_mlp,
        ap_difficulty_layer,
        ap_competence_layer,
        ap_latent_dim,
        n_em_steps=10,
        n_fine_tune_epochs=0,
        warm_start=True,
        optimizer=None,
        optimizer_dict=None,
        lr_scheduler=None,
        lr_scheduler_dict=None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.gt_embed_x = gt_embed_x
        self.gt_mlp = gt_mlp
        self.ap_difficulty_layer = ap_difficulty_layer
        self.ap_competence_layer = ap_competence_layer
        self.ap_latent_dim = ap_latent_dim
        self.n_em_steps = n_em_steps
        self.n_fine_tune_epochs = n_fine_tune_epochs
        self.warm_start = warm_start
        self.optimizer = optimizer
        self.optimizer_dict = optimizer_dict
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_dict = lr_scheduler_dict
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

        # Compute all combinations between annotators and samples.
        a = a[0]
        n_annotators = a.shape[0]
        n_samples = x.shape[0]
        combs = torch.cartesian_prod(torch.arange(n_samples), torch.arange(n_annotators))

        # Compute annotator competences.
        C = self.ap_competence_layer(a)[combs[:, 1]]
        C = C.reshape((-1, self.n_classes, self.n_classes, self.ap_latent_dim))

        # Compute sample difficulties.
        D = self.ap_difficulty_layer(x_learned.detach())[combs[:, 0]]
        D = D.reshape((-1, self.n_classes, self.n_classes, self.ap_latent_dim))

        # Compute annotator performances by combining annotator competences and sample difficulties.
        logits_perf = torch.einsum("ncdl,ncdl->ncd", C, D)
        logits_perf = logits_perf.reshape((n_samples, n_annotators, self.n_classes, self.n_classes))

        return logits_class, logits_perf

    def training_step(self, train_batch, batch_idx):
        x, y, y_est, a = train_batch
        logits_class, logits_perf = self.forward(x, a)
        loss = self._annot_cross_entropy_loss(y=y, y_est=y_est, logits_class=logits_class, logits_perf=logits_perf)
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
        # Setup optimizer(s).
        optimizer = AdamW if self.optimizer is None else self.optimizer
        optimizer_dict = {} if self.optimizer_dict is None else self.optimizer_dict
        optimizer = optimizer(self.parameters(), **optimizer_dict)

        # Return optimizer, if no learning rate scheduler has been defined.
        if self.lr_scheduler is None:
            return [optimizer]

        # Setup learning scheduler(s).
        lr_scheduler_dict = {} if self.lr_scheduler_dict is None else self.lr_scheduler_dict
        lr_scheduler = self.lr_scheduler(optimizer, **lr_scheduler_dict)
        return [optimizer], [lr_scheduler]

    def _annot_cross_entropy_loss(self, y, y_est, logits_class, logits_perf):
        # Obtain predictions of the ground truth and annotator performance model.
        p_class_log = F.log_softmax(logits_class, dim=-1)
        p_perf_log = F.log_softmax(logits_perf, dim=-1)

        # Check whether M-step or fine-tuning is to be performed.
        perform_m_step = self.current_epoch <= self.trainer.max_epochs - self.n_fine_tune_epochs
        if perform_m_step:
            # Perform M-step.
            gt_loss = (-y_est * p_class_log).sum()
            y_one_hot = F.one_hot(y + 1, num_classes=self.n_classes + 1)[:, :, 1:]
            ap_loss = (p_perf_log * y_one_hot[:, :, None, :]).sum(dim=-1).sum(dim=1)
            ap_loss = (-y_est * ap_loss).sum()
            loss = gt_loss + ap_loss
        else:
            # Fine tune by maximizing marginal likelihood.
            p_annot_log = torch.logsumexp(p_class_log[:, None, :, None] + p_perf_log, dim=2).swapaxes(1, 2)
            loss = F.nll_loss(p_annot_log, y, reduction="sum", ignore_index=-1)
        loss /= len(y)
        return loss

    @torch.no_grad()
    def on_train_epoch_start(self):
        # Check whether E-step is to be performed.
        if self.current_epoch % ((self.trainer.max_epochs - self.n_fine_tune_epochs) // self.n_em_steps) == 0:
            # Perform E-step.
            X, A, Y, Y_est = [], [], [], []
            for step, (x, y, _, a) in enumerate(self.trainer.train_dataloader):
                X.append(x)
                Y.append(y)
                y_one_hot = F.one_hot(y + 1, num_classes=self.n_classes + 1)[:, :, 1:]
                if self.trainer.current_epoch == 0:
                    # Initial E-step via soft majority vote.
                    y_est = y_one_hot.sum(dim=1).float()
                    y_est /= y_est.sum(dim=-1, keepdims=True)
                else:
                    # E-step using outputs of ground truth and annotator performance model.
                    y_one_hot = y_one_hot.to(self.device)
                    logits_class, logits_perf = self.forward(x.to(self.device), a.to(self.device))
                    p_class = F.softmax(logits_class, dim=-1)
                    p_perf = F.softmax(logits_perf, dim=-1)
                    ap_prod = (p_perf ** y_one_hot[:, :, None, :]).prod(dim=-1).prod(dim=1)
                    y_est = p_class * ap_prod
                    y_est /= y_est.sum(dim=-1, keepdims=True)
                Y_est.append(y_est.cpu())

            # Update data set.
            self.trainer.train_dataloader.dataset.datasets.X = torch.cat(X, dim=0)
            self.trainer.train_dataloader.dataset.datasets.y = torch.cat(Y, dim=0)
            self.trainer.train_dataloader.dataset.datasets.y_est = torch.cat(Y_est, dim=0)

            # Check whether a warm or cold start is to be performed after each E-step.
            if not self.warm_start:
                # Reset optimizer and potential learning rate schedulers.
                configs = self.configure_optimizers()
                if self.lr_scheduler is None:
                    self.trainer.optimizers = configs
                else:
                    self.trainer.optimizers, self.trainer.lr_schedulers = configs
                if self.trainer.current_epoch > 0:
                    self.apply(self._weight_reset)

    @torch.no_grad()
    def _weight_reset(self, m):
        # Check if the current module has reset_parameters as a callable.
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
