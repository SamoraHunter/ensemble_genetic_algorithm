import argparse
import datetime
import logging
import os
import pathlib
import time
from IPython.display import clear_output
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ml_grid.pipeline import data, main_ga
from ml_grid.util.global_params import global_parameters
from ml_grid.util.grid_param_space_ga import Grid
from ml_grid.util.logger_setup import setup_logger
from ml_grid.util.project_score_save import project_score_save_class
from ml_grid.util.GA_results_explorer import GA_results_explorer
from ml_grid.pipeline.evaluate_methods_ga import get_y_pred_resolver


def main(config_path: str, plot: bool = False, evaluate: bool = False):
    """
    Main function to run the genetic algorithm experiment.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # 1. Initialize global parameters from the specified config file.
    #    We create a temporary instance first to get the original feature names
    #    before any data sampling might occur in the main loop.
    #    The config_path is passed to both global_parameters and Grid.
    global_params = global_parameters(config_path=config_path)

    # 2. Create a unique, timestamped directory for this experiment run.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_specific_dir = os.path.join(global_params.base_project_dir, timestamp)
    pathlib.Path(run_specific_dir).mkdir(parents=True, exist_ok=True)

    # 3. Set up the logger to log into this specific run's directory.
    logger = setup_logger(log_folder_path=run_specific_dir)
    logger.info(f"Using configuration from: {config_path}")
    logger.info(f"Experiment outputs will be saved in: {run_specific_dir}")

    # 4. Update the global_params object to use this new directory for all outputs.
    global_params.base_project_dir = run_specific_dir

    # 5. Initialize the project score CSV file in the run-specific directory.
    project_score_save_class(global_params.base_project_dir)

    # 6. Define the search space for the experiment.
    grid = Grid(
        global_params=global_params,
        test_grid=global_params.testing,
        config_path=config_path,
    )

    # 7. Run the main experiment loop.
    for i in tqdm(range(global_params.n_iter), desc="Grid Search Iterations"):
        # Get the next set of hyperparameters from the grid
        local_param_dict = next(grid.settings_list_iterator)

        # Create the ml_grid_object for this iteration
        ml_grid_object = data.pipe(
            global_params=global_params,
            file_name=global_params.input_csv_path,
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=global_params.base_project_dir,
            param_space_index=i,
            testing=global_params.testing,
        )

        # Run the genetic algorithm with these settings
        main_ga.run(ml_grid_object, local_param_dict=local_param_dict, global_params=global_params).execute()

    # 8. Optionally evaluate the best model and generate plots after the experiment.
    if evaluate:
        logger.info("--- Experiment finished. Starting evaluation... ---")
        try:
            results_path = os.path.join(run_specific_dir, "final_grid_score_log.csv")
            results_df = pd.read_csv(results_path)

            logger.info("--- Evaluating Best Ensemble on Validation Set ---")
            # Find the best run based on the AUC score
            best_run = results_df.loc[results_df['auc'].idxmax()]
            logger.info(f"Best run identified with AUC: {best_run['auc']:.4f}")

            # Re-create the ml_grid_object for the best run's parameters
            ml_grid_object = data.pipe(
                global_params=global_params,
                file_name=global_params.input_csv_path,
                drop_term_list=[],
                local_param_dict=best_run.to_dict(),
                base_project_dir=global_params.base_project_dir,
                param_space_index=best_run.name, # Use index of best run
                testing=global_params.testing,
            )

            # The 'best_ensemble' is stored as a string, so we need to evaluate it
            best_ensemble_str = best_run['best_ensemble']
            best_ensemble = eval(best_ensemble_str)

            # Get predictions on the hold-out validation set
            y_pred_orig = get_y_pred_resolver(
                ensemble=best_ensemble, ml_grid_object=ml_grid_object, valid=True
            )
            y_true_orig = ml_grid_object.y_test_orig

            # Print classification report
            logger.info("\n" + classification_report(y_true_orig, y_pred_orig))

            # Print confusion matrix
            logger.info("Confusion Matrix (Validation Set):")
            logger.info("\n" + str(confusion_matrix(y_true_orig, y_pred_orig)))
        except Exception as e:
            logger.error(f"❌ Failed during evaluation: {e}", exc_info=True)

    if plot:
        logger.info("--- Starting plot generation... ---")
        try:
            results_path = os.path.join(run_specific_dir, "final_grid_score_log.csv")
            results_df = pd.read_csv(results_path)

            # To initialize the explorer, we need a baseline set of feature names.
            initial_data = pd.read_csv(global_params.input_csv_path, nrows=1)
            original_feature_names = list(initial_data.columns)

            # Initialize the results explorer
            explorer = GA_results_explorer(
                df=results_df, original_feature_names=original_feature_names
            )

            # Generate all plots and save them in the experiment directory
            explorer.run_all_plots(plot_dir=run_specific_dir)
            logger.info(f"--- Plots saved to: {run_specific_dir} ---")
        except Exception as e:
            logger.error(f"❌ Failed during plot generation: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Ensemble Genetic Algorithm.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the YAML configuration file (e.g., 'config.yml').",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save analysis plots after the experiment.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the best found ensemble on the validation set after the experiment.",
    )
    args = parser.parse_args()

    main(config_path=args.config, plot=args.plot, evaluate=args.evaluate)