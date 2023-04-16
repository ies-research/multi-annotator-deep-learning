import matplotlib.pyplot as plt
import numpy as np

from skactiveml.utils import ExtLabelEncoder

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)
from sklearn.utils import check_array


def plot_annot_perfs_clf(
    X,
    y_true,
    y_full,
    y,
    clf,
    missing_label=-1,
    figsize=None,
    filepath=None,
    cmap_clf="seismic",
    cmap_annot="bone",
    markersize=100,
    labelsize=15,
    A=None,
    plot_colorbar=True,
    plot_accuracies=True,
):
    """
    Visualizes the predicted class-membership probabilities and annotator performances.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Two-dimensional samples.
    y_true : array-like of shape (n_samples)
        True class labels of the samples `X`.
    y_full : array-like of shape (n_samples, n_annotators)
        Class labels provided by annotators excluding any missing labels.
    y : array-like of shape (n_samples, n_annotators)
        Class labels provided by the annotators including missing labels.
    clf : sklearn.base.ClassifierMixin and AnnotatorMixin
        Trained classifier and annotator performance model implementing
        `predict_proba` and `predict_annotator_perf`.
    missing_label : scalar or string or np.nan or None, optional (default=-1)
        Value to represent a missing label.
    figsize : tuple of shape (width, height), optional (default=(6, 8))
        Describes the width and height of each figure.
    filepath : string, optional (default=None)
        Location of the figures to be saved.
    cmap_clf : string, optional (default='bone')
        Name of colormap to plot probabilities predicted by the classifier.
    cmap_annot : string, optional (default='seismic')
        Name of colormap to plot predicted annotator performances.
    markersize : float, optional (default=100)
        Size of the markers to be plotted as samples.
    labelsize : float, optional (default=15)
        Fontsize of the labels and ticks of the figure's axes.
    plot_colorbar : bool, optional (default=True)
        Flag indicating whether the colorbar of the predictions is to be plotted.
    plot_accuracies : bool, optional (default=True)
        Flag indicating whether the accuracies of the individual annotators are to be plotted.
    """
    le = ExtLabelEncoder(missing_label=missing_label, classes=[0, 1]).fit(y_true)
    y_true = le.transform(y_true)
    y = le.transform(y)
    xx = np.linspace(np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5, 100)
    yy = np.linspace(np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5, 100)
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # Compute performance of GT model for classification.
    P_mesh = clf.predict_proba(Xfull, data_loader_dict={"batch_size": 256})
    y_pred = P_mesh.argmax(axis=-1)

    # Compute performance of AP model for classification.
    n_annotators = y.shape[1]
    if A is not None:
        P_ap_mesh = clf.predict_annotator_perf(Xfull, A=A, data_loader_dict={"batch_size": 256})
    else:
        P_ap_mesh = clf.predict_annotator_perf(Xfull, data_loader_dict={"batch_size": 256})
    y_pred_ap = clf.predict_annotator_perf(X, data_loader_dict={"batch_size": 256}) > 0.5
    y_true_ap = np.equal(y_full, np.tile(y_true, (n_annotators, 1)).T)

    # Plot setup.
    figsize = (8, 6) if figsize is None else figsize
    levels = np.linspace(0, 1, 25)
    for i in range(n_annotators + 1):
        f, ax = plt.subplots(figsize=figsize)
        if i == 0:
            cmap = cmap_clf
            if plot_accuracies:
                bal_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
                acc = accuracy_score(y_true=y_true, y_pred=y_pred)
                ax.set_title(
                    "GT model: "
                    + "\n micro-accuracy: {:0.2f}".format(acc)
                    + "\n macro-accuracy: {:0.2f}".format(bal_acc)
                )
            probs = P_mesh[:, 1].reshape(xx.shape)
            ax.scatter(
                X[:, 0],
                X[:, 1],
                c=y_true,
                edgecolor="k",
                s=markersize,
                linewidth=0.5,
                zorder=2,
                cmap=cmap_clf,
            )
        else:
            cmap = cmap_annot
            not_nan_a = ~(y[:, i - 1] == -1)
            y_a = y[not_nan_a, i - 1]
            X_a = X[not_nan_a]
            if plot_accuracies:
                bal_acc_ap = balanced_accuracy_score(y_true=y_true_ap[:, i - 1], y_pred=y_pred_ap[:, i - 1])
                acc_ap = accuracy_score(y_true=y_true_ap[:, i - 1], y_pred=y_pred_ap[:, i - 1])
                ax.set_title(
                    "AP model: "
                    + "\n micro-accuracy: {:0.2f}".format(acc_ap)
                    + "\n macro-accuracy: {:0.2f}".format(bal_acc_ap)
                )
            probs = P_ap_mesh[:, i - 1].reshape(xx.shape)
            z = y_true[not_nan_a] == y_a
            ax.scatter(
                X_a[z, 0],
                X_a[z, 1],
                marker="o",
                edgecolor="k",
                linewidth=0.5,
                cmap=cmap_clf,
                zorder=2,
                s=markersize,
                c=y_a[z],
                vmin=0,
                vmax=1,
                linewidths=0.5,
            )
            ax.scatter(
                X_a[~z, 0],
                X_a[~z, 1],
                marker="X",
                edgecolor="k",
                linewidth=1,
                cmap=cmap_clf,
                zorder=2,
                s=markersize,
                c=y_a[~z],
                vmin=0,
                vmax=1,
                linewidths=0.5,
            )
        probs = np.round(probs, 2)
        contour = ax.contourf(xx, yy, probs, levels=levels, cmap=cmap, zorder=0, alpha=0.6)
        if i == 0:
            ax.contour(xx, yy, probs, levels=[0.5], colors=["k"], zorder=1)
        if plot_colorbar:
            ax_c = f.colorbar(contour)
            ticks = [0, 0.25, 0.5, 0.75, 1]
            ax_c.set_ticks(ticks)
            ax_c.set_ticklabels(ticks)
            ax_c.ax.tick_params(labelsize=labelsize)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if filepath is not None:
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(f"{filepath}-{i}.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()


def plot_annot_cohen_kappa_scores(y, missing_label=-1, filepath=None):
    """
    Plots the Cohen's kappa score for each pair of annotators.

    Parameters
    ----------
    y : array-like of shape (n_samples, n_annotators)
        Class labels provided by the annotators including missing labels.
    missing_label : scalar or string or np.nan or None, optional (default=-1)
    filepath : string, optional (default=None)
        Location for saving figure.

    Returns
    -------
    M : numpy.narray of shape (n_annotators, n_annotators)
        `M[i,j]` indicates Cohen's kappa score of annotator `i` and `j`.
    """
    n_annotators = np.size(y, axis=1)
    y = check_array(y, force_all_finite=False)
    y = ExtLabelEncoder(missing_label=missing_label).fit_transform(y)
    M = np.zeros((n_annotators, n_annotators))
    for i in range(n_annotators):
        for j in range(n_annotators):
            is_labeled = np.logical_and(~(y[:, i] == -1), ~(y[:, j] == -1))
            M[i, j] = cohen_kappa_score(y[is_labeled][:, i], y[is_labeled][:, j])
    _plot_annot_interdependencies(M=M, filename=filepath)
    return M


def _plot_annot_interdependencies(M, filename=None):
    f, ax = plt.subplots(figsize=(8, 6))
    ticks = [0, 0.25, 0.5, 0.75, 1]
    labelsize = 15
    img = ax.imshow(M, cmap="binary_r")
    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    ax.tick_params(axis="both", which="minor", labelsize=labelsize)
    ax.set(xlabel="annotator", ylabel="annotator")

    ax_c = f.colorbar(img)
    ax_c.set_ticks(ticks)
    ax_c.set_ticklabels(ticks)
    ax_c.ax.tick_params(labelsize=labelsize)

    if filename is not None:
        plt.savefig(f"{filename}.pdf", rasterized=True)
    plt.show()
