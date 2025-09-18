import matplotlib.pyplot as plt
from typing import Dict, List, Union
import pandas as pd

# class plot_methods():

# def __init__(self):


def plot_pie_chart_with_counts(
    X_train: pd.DataFrame, X_test: pd.DataFrame, X_test_orig: pd.DataFrame
) -> None:
    """Plots a pie chart showing the relative sizes of data splits.

    This function visualizes the distribution of samples across the training,
    testing, and original testing (validation) sets.

    Args:
        X_train: The training dataset DataFrame.
        X_test: The test dataset DataFrame.
        X_test_orig: The original test (validation) dataset DataFrame.
    """
    sizes = [len(X_train), len(X_test), len(X_test_orig)]
    labels = ["X_train", "X_test", "X_test_orig"]

    # Colors for the pie chart sections
    colors = ["#ff9999", "#66b3ff", "#c2c2f0"]

    # Explode the section with the largest size
    explode = tuple(
        0.1 if i == sizes.index(max(sizes)) else 0 for i in range(len(sizes))
    )

    # Create the pie chart
    plt.figure(figsize=(2, 2))
    patches, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        explode=explode,
    )

    # Equal aspect ratio ensures that the pie chart is drawn as a circle.
    plt.axis("equal")

    # Add a title
    plt.title("Sizes of Datasets")

    # Add value counts to the plot
    for i, text in enumerate(texts):
        percent = sizes[i] / sum(sizes) * 100
        text.set_text(f"{labels[i]}: {sizes[i]} ({percent:.1f}%)")

    plt.show()


def plot_dict_values(data_dict: Dict[str, bool]) -> None:
    """Creates a horizontal bar chart to visualize boolean values in a dictionary.

    This function is useful for displaying configuration flags, where each key
    is a setting and its value is either True (green) or False (red).

    Args:
        data_dict: A dictionary where keys are strings and values are booleans.
    """
    # Extract the keys and values from the dictionary
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Define colors for True and False values
    colors = ["green" if val else "red" for val in values]

    # Create the horizontal bar chart
    plt.figure(figsize=(2, 2))
    plt.barh(keys, [1] * len(keys), color=colors)
    plt.yticks(rotation=0)  # Keep y-axis ticks horizontal

    # Add a legend for the colors
    legend_colors = [
        plt.Rectangle((0, 0), 1, 1, fc="green", edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor="none"),
    ]
    plt.legend(legend_colors, ["True", "False"], loc="upper right")

    # Set axis labels and title
    plt.xlabel("Value")
    plt.ylabel("Data Fields")
    plt.title("Values for the Given Dictionary")

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Given dictionary


# def create_bar_chart(data_dict, title='', x_label='', y_label=''):
#     # Extracting keys and values from the dictionary
#     categories = list(data_dict.keys())
#     values = list(data_dict.values())

#     # Setting up the bar chart
#     plt.bar(categories, values)

#     # Adding labels and title
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)

#     # Displaying the chart
#     plt.show()


def create_bar_chart(
    data_dict: Dict[str, int], title: str = "", x_label: str = "", y_label: str = ""
) -> None:
    """Creates a horizontal bar chart from a dictionary of counts.

    Args:
        data_dict: A dictionary where keys are categories (str) and values
            are their counts (int).
        title: The title for the chart.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
    """
    # Extracting keys and values from the dictionary
    categories = list(data_dict.keys())
    values = list(data_dict.values())

    # Setting up the horizontal bar chart
    plt.barh(categories, values)

    # Adding labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Adding angled labels
    for index, value in enumerate(values):
        plt.text(
            value,
            index,
            str(value),
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    # Displaying the chart
    plt.show()


def plot_candidate_feature_category_lists(data: Dict[str, int]) -> None:
    """A wrapper function to plot feature category counts using a bar chart.

    Args:
        data: A dictionary where keys are feature category names and values
            are the number of features in that category.
    """
    create_bar_chart(
        data, title="Feature category counts", x_label="features", y_label="counts"
    )
