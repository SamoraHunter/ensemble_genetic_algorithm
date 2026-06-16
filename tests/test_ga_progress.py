"""Tests for ga_plots.ga_progress module."""

from unittest.mock import MagicMock, patch

import numpy as np

from ml_grid.ga_functions.ga_plots import ga_progress


def test_plot_generation_progress_fitness_basic():
    """Test plot_generation_progress_fitness runs without exception."""
    generation_progress_list = [0.5, 0.6, 0.7, 0.8]

    # Create mock objects for subplots
    mock_fig = MagicMock()
    mock_ax = MagicMock()

    with patch("ml_grid.ga_functions.ga_plots.ga_progress.plt") as mock_plt:
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        ga_progress.plot_generation_progress_fitness(
            generation_progress_list=generation_progress_list,
            pop_val=10,
            g_val=5,
            nb_val=3,
            file_path="/fake/path",
        )

        # Verify plot was created and saved
        assert mock_plt.subplots.called
        assert mock_plt.savefig.called


def test_plot_generation_progress_fitness_single_value():
    """Test plot_generation_progress_fitness with single value."""
    generation_progress_list = [0.7]

    mock_fig = MagicMock()
    mock_ax = MagicMock()

    with patch("ml_grid.ga_functions.ga_plots.ga_progress.plt") as mock_plt:
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        ga_progress.plot_generation_progress_fitness(
            generation_progress_list=generation_progress_list,
            pop_val=5,
            g_val=1,
            nb_val=2,
            file_path="/fake/path",
        )

        assert mock_plt.subplots.called


def test_plot_generation_progress_fitness_uses_numpy_polyfit():
    """Test that plot_generation_progress_fitness uses np.polyfit."""
    generation_progress_list = [0.5, 0.6, 0.7]

    mock_fig = MagicMock()
    mock_ax = MagicMock()

    with patch("ml_grid.ga_functions.ga_plots.ga_progress.plt") as mock_plt:
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Make polyfit return a known slope
        with patch("numpy.polyfit", return_value=np.array([0.1, 0.5])) as mock_polyfit:
            ga_progress.plot_generation_progress_fitness(
                generation_progress_list=generation_progress_list,
                pop_val=5,
                g_val=1,
                nb_val=2,
                file_path="/fake/path",
            )

            # Verify polyfit was called with correct arguments
            assert mock_polyfit.called
            call_args = mock_polyfit.call_args[0]
            assert len(call_args) == 3  # x, y, deg


def test_plot_generation_progress_fitness_line_of_best_fit():
    """Test that line of best fit is computed and plotted."""
    generation_progress_list = [0.5, 0.6, 0.7]

    mock_fig = MagicMock()
    mock_ax = MagicMock()

    with patch("ml_grid.ga_functions.ga_plots.ga_progress.plt") as mock_plt:
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Make polyfit return a known slope
        with patch("numpy.polyfit", return_value=np.array([0.1, 0.5])):
            ga_progress.plot_generation_progress_fitness(
                generation_progress_list=generation_progress_list,
                pop_val=5,
                g_val=1,
                nb_val=2,
                file_path="/fake/path",
            )

            # Verify line_of_best_fit was created
            assert mock_plt.savefig.called
