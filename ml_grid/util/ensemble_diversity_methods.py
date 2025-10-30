from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.spatial.distance


class EnsembleDiversityMeasurer:
    """Calculates the diversity of an ensemble of classifiers using various metrics.

    This class provides several methods to quantify the diversity among the
    predictions of base learners in an ensemble. A higher diversity score
    generally indicates that the models are making different errors, which can
    lead to a more robust final ensemble.
    """

    method: str
    """The diversity measurement method to use."""

    weights: List[float]
    """A list of weights for combining metrics in 'comprehensive' mode."""

    def __init__(
        self, method: str = "comprehensive", weights: Optional[List[float]] = None
    ):
        """Initializes the EnsembleDiversityMeasurer.

        Args:
            method: The diversity measurement method. Can be "jaccard",
                "hamming", "disagreement", "q_statistic", "kappa", or
                "comprehensive". Defaults to "comprehensive".
            weights: A list of four weights for the 'comprehensive' method,
                corresponding to [jaccard, hamming, disagreement, q_stat].
                Defaults to equal weights [0.25, 0.25, 0.25, 0.25].
        """
        self.method = method
        self.weights = weights or [0.25, 0.25, 0.25, 0.25]  # Equal weights by default

    def measure_jaccard_diversity(self, ensemble: List[Any]) -> float:
        """Calculates diversity using the average Jaccard distance.

        Jaccard distance measures the dissimilarity between sample sets. A higher
        value indicates greater diversity.

        Args:
            ensemble: A list containing the ensemble configuration. The predictions
                are expected at `ensemble[0][i][5]`.

        Returns:
            The average Jaccard distance between all pairs of prediction vectors,
            or 0.0 if calculation is not possible.
        """
        n_y_pred = len(ensemble[0])

        if n_y_pred < 2:
            return 0.0

        all_y_pred_arrays = []
        for i in range(n_y_pred):
            pred_array = np.array(ensemble[0][i][5], dtype=int)
            all_y_pred_arrays.append(pred_array)

        try:
            distance_vector = scipy.spatial.distance.pdist(
                all_y_pred_arrays, metric="jaccard"
            )
            return np.mean(distance_vector) # type: ignore
        except Exception:
            return 0.0

    def measure_hamming_diversity(self, ensemble: List[Any]) -> float:
        """Calculates diversity using the average Hamming distance.

        Hamming distance measures the proportion of positions at which two
        prediction vectors differ.

        Args:
            ensemble: A list containing the ensemble configuration. The predictions
                are expected at `ensemble[0][i][5]`.

        Returns:
            The average Hamming distance between all pairs of prediction vectors,
            or 0.0 if calculation is not possible.
        """
        n_y_pred = len(ensemble[0])

        if n_y_pred < 2:
            return 0.0

        all_y_pred_arrays = []
        for i in range(n_y_pred):
            pred_array = np.array(ensemble[0][i][5], dtype=int)
            all_y_pred_arrays.append(pred_array)

        try:
            distance_vector = scipy.spatial.distance.pdist(
                all_y_pred_arrays, metric="hamming"
            )
            return np.mean(distance_vector) # type: ignore
        except Exception:
            return 0.0

    def measure_disagreement_diversity(self, ensemble: List[Any]) -> float:
        """Calculates diversity based on the variance of predictions per instance.

        This method measures the average disagreement among classifiers for each
        data point. A higher value indicates more disagreement and thus more
        diversity.

        Args:
            ensemble: A list containing the ensemble configuration. The predictions
                are expected at `ensemble[0][i][5]`.

        Returns:
            The mean disagreement, normalized by the maximum possible disagreement.
        """
        n_y_pred = len(ensemble[0])

        if n_y_pred < 2:
            return 0.0

        try:
            predictions = np.array([ensemble[0][i][5] for i in range(n_y_pred)])

            disagreements = []
            for instance_idx in range(predictions.shape[1]):
                instance_preds = predictions[:, instance_idx]
                disagreement = np.var(instance_preds)
                disagreements.append(disagreement)

            # Maximum possible disagreement for binary is 0.25
            max_disagreement = 0.25
            return np.mean(disagreements) / max_disagreement # type: ignore
        except Exception:
            return 0.0

    def measure_q_statistic_diversity(self, ensemble: List[Any]) -> float:
        """Calculates diversity using Yule's Q statistic.

        The Q statistic measures the pairwise association between classifiers.
        A value near 0 indicates independence (high diversity), while a value
        near 1 indicates strong positive correlation (low diversity). This
        method returns `1 - mean(abs(Q))`.

        Args:
            ensemble: A list containing the ensemble configuration. The predictions
                are expected at `ensemble[0][i][5]`.

        Returns:
            The diversity score based on the Q statistic.
        """
        n_y_pred = len(ensemble[0])

        if n_y_pred < 2:
            return 0.0

        try:
            predictions = np.array([ensemble[0][i][5] for i in range(n_y_pred)])
            q_statistics = []

            for i in range(n_y_pred):
                for j in range(i + 1, n_y_pred):
                    pred_i = predictions[i]
                    pred_j = predictions[j]

                    n11 = np.sum((pred_i == 1) & (pred_j == 1))
                    n10 = np.sum((pred_i == 1) & (pred_j == 0))
                    n01 = np.sum((pred_i == 0) & (pred_j == 1))
                    n00 = np.sum((pred_i == 0) & (pred_j == 0))

                    denominator = (n11 * n00) + (n01 * n10)
                    if denominator > 0:
                        q = ((n11 * n00) - (n01 * n10)) / denominator
                        q_statistics.append(abs(q))

            if not q_statistics:
                return 0.0

            return 1 - np.mean(q_statistics) # type: ignore
        except Exception:
            return 0.0

    def measure_kappa_diversity(self, ensemble: List[Any]) -> float:
        """Calculates diversity using Cohen's Kappa statistic.

        Kappa measures the agreement between two classifiers, corrected for
        chance agreement. A value near 0 indicates agreement is at chance level
        (high diversity), while a value near 1 indicates perfect agreement
        (low diversity). This method returns `1 - mean(abs(kappa))`.

        Args:
            ensemble: A list containing the ensemble configuration. The predictions
                are expected at `ensemble[0][i][5]`.

        Returns:
            The diversity score based on the Kappa statistic.
        """
        n_y_pred = len(ensemble[0])

        if n_y_pred < 2:
            return 0.0

        try:
            predictions = np.array([ensemble[0][i][5] for i in range(n_y_pred)])
            kappa_values = []

            for i in range(n_y_pred):
                for j in range(i + 1, n_y_pred):
                    pred_i = predictions[i]
                    pred_j = predictions[j]

                    # Calculate observed agreement
                    agreement = np.mean(pred_i == pred_j)

                    # Calculate expected agreement
                    p1_i = np.mean(pred_i)  # P(pred_i = 1)
                    p1_j = np.mean(pred_j)  # P(pred_j = 1)

                    expected_agreement = (p1_i * p1_j) + ((1 - p1_i) * (1 - p1_j))

                    # Calculate kappa
                    if expected_agreement < 1.0:
                        kappa = (agreement - expected_agreement) / (
                            1 - expected_agreement
                        )
                        kappa_values.append(abs(kappa))

            if not kappa_values:
                return 0.0

            return 1 - np.mean(kappa_values) # type: ignore
        except Exception:
            return 0.0

    def measure_binary_vector_diversity(self, ensemble: List[Any]) -> float:
        """Measures ensemble diversity based on the configured method.

        This is the main entry point for measuring diversity. It calls the
        appropriate measurement function based on `self.method`.

        Args:
            ensemble: A list containing the ensemble configuration.

        Returns:
            The calculated diversity score, where 0 is identical and 1 is
            maximally diverse.
        """
        if self.method == "jaccard":
            return self.measure_jaccard_diversity(ensemble)
        elif self.method == "hamming":
            return self.measure_hamming_diversity(ensemble)
        elif self.method == "disagreement":
            return self.measure_disagreement_diversity(ensemble)
        elif self.method == "q_statistic":
            return self.measure_q_statistic_diversity(ensemble)
        elif self.method == "kappa":
            return self.measure_kappa_diversity(ensemble)
        elif self.method == "comprehensive":
            # Combine all methods
            jaccard_div = self.measure_jaccard_diversity(ensemble)
            hamming_div = self.measure_hamming_diversity(ensemble)
            disagreement_div = self.measure_disagreement_diversity(ensemble)
            q_stat_div = self.measure_q_statistic_diversity(ensemble)

            # Weighted combination
            combined_diversity = (
                self.weights[0] * jaccard_div
                + self.weights[1] * hamming_div
                + self.weights[2] * disagreement_div
                + self.weights[3] * q_stat_div
            )

            return combined_diversity
        else:
            raise ValueError(f"Unknown method: {self.method}")


def measure_diversity_wrapper(
    individual: List[Any], method: str = "comprehensive"
) -> float:
    """A wrapper function to measure ensemble diversity.

    This function instantiates `EnsembleDiversityMeasurer` and calls its main
    measurement method.

    Args:
        individual: The ensemble data structure.
        method: The diversity measurement method to use. Defaults to "comprehensive".

    Returns:
        The calculated diversity score.
    """
    measurer = EnsembleDiversityMeasurer(method=method)
    return measurer.measure_binary_vector_diversity(individual)


def apply_diversity_penalty(
    auc: float, mcc: float, diversity_metric: float, diversity_params: Dict
) -> Tuple[float, float]:
    """Applies a penalty to performance metrics based on ensemble diversity.

    Args:
        auc: The original AUC score.
        mcc: The original MCC score.
        diversity_metric: The diversity score (0=identical, 1=diverse).
        diversity_params: A dictionary with penalty parameters, including
            'penalty_method' and 'penalty_strength'.

    Returns:
        A tuple containing the penalized AUC and MCC scores.
    """
    penalty_method = diversity_params.get("penalty_method", "linear")
    penalty_strength = diversity_params.get("penalty_strength", 0.3)
    min_score_factor = diversity_params.get("min_score_factor", 0.1)

    # Convert diversity to similarity (1 = similar/bad, 0 = diverse/good)
    similarity_metric = max(0, 1 - diversity_metric)

    if penalty_method == "linear":
        # Linear penalty
        diversity_factor = 1 - (penalty_strength * similarity_metric)

    elif penalty_method == "quadratic":
        # Quadratic penalty - gentler at first, harsher for very similar
        diversity_factor = 1 - (penalty_strength * similarity_metric**2)

    elif penalty_method == "exponential":
        # Exponential penalty - more aggressive
        diversity_factor = np.exp(-penalty_strength * similarity_metric)

    elif penalty_method == "threshold":
        # Only penalize if similarity above threshold
        similarity_threshold = diversity_params.get("similarity_threshold", 0.7)
        if similarity_metric > similarity_threshold:
            excess_similarity = similarity_metric - similarity_threshold
            diversity_factor = 1 - (penalty_strength * excess_similarity)
        else:
            diversity_factor = 1.0
    else:
        diversity_factor = 1.0

    # Apply penalty with minimum score protection
    diversity_factor = max(min_score_factor, diversity_factor)

    penalized_auc = auc * diversity_factor
    penalized_mcc = mcc * diversity_factor

    return penalized_auc, penalized_mcc
