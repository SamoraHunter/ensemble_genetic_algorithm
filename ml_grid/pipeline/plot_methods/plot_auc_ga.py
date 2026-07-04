from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_auc_base(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    fig: Optional[plt.Figure] = None,
    plot_dir: Optional[str] = None,
) -> plt.Figure:
    """Creates a ROC curve plot without displaying it.

    Args:
        y_true: The ground truth binary labels.
        y_pred: The predicted probabilities or decision function scores.
        title: A title for the plot, often used to identify the model or run.
        fig: An optional figure object to use. If None, a new figure is created.
            Defaults to None.
        plot_dir: The directory to save the plot image. If None, the plot is
            not saved. Defaults to None.

    Returns:
        The matplotlib figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    if fig is None:
        fig = plt.figure(figsize=(8, 6))
    else:
        fig.clear()

    ax = fig.add_subplot(111)
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Receiver Operating Characteristic: {title}")
    ax.legend(loc="lower right")

    if plot_dir is not None:
        fig.savefig(plot_dir)

    return fig


def plot_auc(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, plot_dir: Optional[str] = None
) -> None:
    """Computes and plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true: The ground truth binary labels.
        y_pred: The predicted probabilities or decision function scores.
        title: A title for the plot, often used to identify the model or run.
        plot_dir: The directory to save the plot image. If None, the plot is
            only displayed. Defaults to None.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic: {title}")
    plt.legend(loc="lower right")

    if plot_dir is not None:
        plt.savefig(plot_dir)

    plt.show()
    plt.close()
