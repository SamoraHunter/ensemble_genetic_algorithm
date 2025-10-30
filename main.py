import argparse
import datetime
import logging
import os
import pathlib
from ml_grid.util.logger_setup import setup_logger, get_logger
from ml_grid.util.project_score_save import project_score_save_class
from ml_grid.util.GA_results_explorer import GA_results_explorer
from ml_grid.pipeline.evaluate_methods_ga import get_y_pred_resolver
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from ml_grid.pipeline import data, main_ga
from ml_grid.util.global_params import global_parameters
from ml_grid.util.grid_param_space_ga import Grid


def initialize_logger(config_path: str) -> logging.Logger:
    """
    Initializes and returns a logger for the experiment.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_specific_dir = os.path.join("experiments", timestamp)  # Store experiments in a dedicated folder
    pathlib.Path(run_specific_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_folder_path=run_specific_dir)
    logger.info(f"Using configuration from: {config_path}")
    logger.info(f"Experiment outputs will be saved in: {run_specific_dir}")
    return logger

def main(config_path: str, plot: bool = False, evaluate: bool = False):
    """
    Main function to run the genetic algorithm experiment.

    Args:
        config_path (str): Path to the YAML configuration file.
        plot (bool): Whether to generate and save analysis plots.
        evaluate (bool): Whether to evaluate the best ensemble on the validation set.
    """
    logger = initialize_logger(config_path)

    # 1. Initialize global parameters from the specified config file.
    global_params = global_parameters(config_path=config_path)

    # 2. Update the base project directory to the run-specific directory.
    run_specific_dir = logger.handlers[0].baseFilename.rsplit('/', 1)[0]
    global_params.base_project_dir = run_specific_dir

    # 3. Initialize the project score CSV file.
    project_score_save_class(global_params.base_project_dir)

    # 4. Define the search space for the experiment.
    grid = Grid(
        global_params=global_params,
        test_grid=global_params.testing,
        config_path=config_path,
    )

    # 5. Run the main experiment loop.
    for i in tqdm(range(global_params.n_iter), desc="Grid Search Iterations"):
        local_param_dict = next(grid.settings_list_iterator)

        ml_grid_object = data.pipe(
            global_params=global_params,
            file_name=global_params.input_csv_path,
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=global_params.base_project_dir,
            param_space_index=i,
            testing=global_params.testing,
        )

        main_ga.run(ml_grid_object, local_param_dict=local_param_dict, global_params=global_params).execute()

    # 6. Optionally evaluate the best model and generate plots.
    if evaluate:
        logger.info("--- Experiment finished. Starting evaluation... ---")
        try:
            results_path = os.path.join(run_specific_dir, "final_grid_score_log.csv")
            results_df = pd.read_csv(results_path)

            logger.info("--- Evaluating Best Ensemble on Validation Set ---")
            best_run = results_df.loc[results_df['auc'].idxmax()]
            logger.info(f"Best run identified with AUC: {best_run['auc']:.4f}")

            ml_grid_object = data.pipe(
                global_params=global_params,
                file_name=global_params.input_csv_path,
                drop_term_list=[],
                local_param_dict=best_run.to_dict(),
                base_project_dir=global_params.base_project_dir,
                param_space_index=best_run.name,
                testing=global_params.testing,
            )

            best_ensemble = eval(best_run['best_ensemble'])

            y_pred_orig = get_y_pred_resolver(
                ensemble=best_ensemble, ml_grid_object=ml_grid_object, valid=True
            )
            y_true_orig = ml_grid_object.y_test_orig

            logger.info("\n" + classification_report(y_true_orig, y_pred_orig))
            logger.info("Confusion Matrix (Validation Set):")
            logger.info("\n" + str(confusion_matrix(y_true_orig, y_pred_orig)))
        except Exception as e:
            logger.error(f"❌ Failed during evaluation: {e}", exc_info=True)

    if plot:
        logger.info("--- Starting plot generation... ---")
        try:
            results_path = os.path.join(run_specific_dir, "final_grid_score_log.csv")
            results_df = pd.read_csv(results_path)

            initial_data = pd.read_csv(global_params.input_csv_path, nrows=1)
            original_feature_names = list(initial_data.columns)

            explorer = GA_results_explorer(
                df=results_df, original_feature_names=original_feature_names
            )

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
