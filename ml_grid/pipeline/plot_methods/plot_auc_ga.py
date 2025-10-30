from typing import Optional

import numpy as np


def plot_auc(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, plot_dir: Optional[str] = None
) -> None:
    """Computes and plots the Receiver Operating Characteristic (ROC) curve.

    Note:
        This function is currently unimplemented. A standard implementation is
        provided in the comments below as a guide for future development. The
        function signature has been updated to reflect the necessary arguments
        for plotting an ROC curve.

    Args:
        y_true: The ground truth binary labels.
        y_pred: The predicted probabilities or decision function scores.
        title: A title for the plot, often used to identify the model or run.
        plot_dir: The directory to save the plot image. If None, the plot is
            only displayed. Defaults to None.
    """
    # The original function is unimplemented. A standard implementation would be:
    #
    # fpr, tpr, _ = roc_curve(y_true, y_pred)
    # roc_auc = auc(fpr, tpr)
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'Receiver Operating Characteristic: {title}')
    # plt.legend(loc="lower right")
    # plt.show()

    return None
