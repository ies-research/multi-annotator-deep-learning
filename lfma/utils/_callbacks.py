import torch

from copy import deepcopy

from pytorch_lightning.callbacks import Callback, TQDMProgressBar

from tqdm import tqdm


class LitProgressBar(TQDMProgressBar):
    """LitProgressBar

    Simplified bar as callback to report the training progress.
    """

    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


class StoreBestModuleStateDict(Callback):
    """StoreBestModuleStateDict

    This class is a simple callback, which is called after each validation step. The goal is to save the `state_dict`
    of the best model found during training.

    Parameters
    ----------
    score_name : str, optional (default="val_acc")
        Name of the score as criterion for assessing model performance.
    maximize : bool, optional (default=True)
        Flag whether the score is to be maximized (`maximize=True`) or minimized (`maximize=False`).
    last : bool, optional (default=False)
        Flag to decide which model is to be saved in case of equal performance. If `last=True`, the newer`state_dict`
        of the new model is stored, otherwise the `state_dict` of the older model is kept.
    """

    def __init__(self, score_name="val_acc", maximize=True, last="False"):
        self.score_name = score_name
        self.maximize = maximize
        self.best_score = -torch.inf if self.maximize else torch.inf
        self.best_model_state_dict = None
        self.last = last

    def on_validation_end(self, trainer, pl_module):
        score = trainer.callback_metrics[self.score_name]
        if self.last:
            if (score >= self.best_score and self.maximize) or (score <= self.best_score and not self.maximize):
                self.best_model_state_dict = deepcopy(pl_module.state_dict())
                self.best_score = score
        else:
            if (score > self.best_score and self.maximize) or (score < self.best_score and not self.maximize):
                self.best_model_state_dict = deepcopy(pl_module.state_dict())
                self.best_score = score
