import numpy as np
import matplotlib.pyplot as plt


def plot_generation_progress_fitness(
    generation_progress_list, pop_val, g_val, nb_val, file_path
):
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
    plt.savefig(
        file_path
        + "/logs/figures/best_pop={}_g={}_nb={}.png".format(pop_val, g_val, nb_val),
        bbox_inches="tight",
    )

    plt.show()  # This shows the plot if you're running the script directly


# Example usage:
# plot_generation_progress_fitness(generation_progress_list, pop_val, g_val, nb_val, file_path)
