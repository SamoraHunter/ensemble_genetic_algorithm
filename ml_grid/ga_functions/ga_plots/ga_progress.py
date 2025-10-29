"""Module for plotting the progress of the genetic algorithm's fitness."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_generation_progress_fitness(
    generation_progress_list: List[float],
    pop_val: int,
    g_val: int,
    nb_val: int,
    file_path: str,
) -> None:
    """Plots the fitness progress across generations and saves the plot.

    This function creates a scatter plot of the fitness scores for each
    generation and overlays a line of best fit to visualize the trend.
    The plot is then saved to a file.

    Args:
        generation_progress_list: A list of fitness scores for each generation.
        pop_val: The population size used in the genetic algorithm.
        g_val: The current generation number.
        nb_val: The number of neighbors considered.
        file_path: The base path to save the plot to.

    Returns:
        None
    """
    fig, ax = plt.subplots()

    # Convert x-axis to integers representing epochs
    x = list(range(1, len(generation_progress_list) + 1))

    ax.scatter(x, generation_progress_list)

    # Calculate the line of best fit
    coeffs = np.polyfit(x, generation_progress_list, 1)
    line_of_best_fit = np.poly1d(coeffs)
    ax.plot(
        x, line_of_best_fit(x), color="red", linestyle="--", label="Line of Best Fit"
    )

    # Set plot title and axis labels
    ax.set_title("Generation Progress Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Evaluation Metric Score")

    # Add legend
    ax.legend()

    # Saving the figure
    save_path = (
        f"{file_path}/logs/figures/best_pop={pop_val}_g={g_val}_nb={nb_val}.png"
    )
    plt.savefig(save_path, bbox_inches="tight")

    plt.show()  # This shows the plot if you're running the script directly