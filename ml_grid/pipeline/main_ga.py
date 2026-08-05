import datetime
import gc
import itertools
import logging
import os
import pathlib
import pickle
import random
import shutil
import time
import traceback
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import tqdm
from deap import base, creator, tools
from IPython.display import clear_output
from sklearn import metrics

# from ml_grid.ga_functions.ga_plots.ga_progress import plot_generation_progress_fitness
from ml_grid.ga_functions.ga_plots.ga_progress import (
    plot_generation_progress_fitness,
    plot_generation_progress_fitness_wrapper,
)
from ml_grid.pipeline.crossover_methods import (
    cxBlend,
    cxOnePoint,
    cxOrdered,
    cxUniform,
)
from ml_grid.pipeline.ensemble_generator_ga import ensembleGenerator
from ml_grid.pipeline.evaluate_methods_ga import (
    evaluate_weighted_ensemble_auc,
    get_y_pred_resolver,
    measure_binary_vector_diversity,
)
from ml_grid.pipeline.mutate_methods import mutateEnsemble
from ml_grid.pipeline.plot_methods.plot_auc_ga import plot_auc, plot_auc_base

# from ml_grid.model_classes import LogisticRegression_class
# from ml_grid.pipeline import grid_search_cross_validate
from ml_grid.util.global_params import global_parameters
from ml_grid.util.grid_param_space_ga import Grid
from ml_grid.util.project_score_save import project_score_save_class

logger = logging.getLogger("ensemble_ga")


class run:
    """Orchestrates the main Genetic Algorithm (GA) evolution process.

    This class is the primary engine for running the ensemble evolution. It takes
    a configuration, sets up the GA environment using the DEAP library, and
    executes the evolutionary loop for a specified number of generations.

    The process for each run includes:
    1.  Initializing the GA parameters (population size, generations, etc.).
    2.  Registering the necessary genetic operators (evaluation, crossover,
        mutation, selection) with the DEAP toolbox.
    3.  Creating an initial population of candidate ensembles.
    4.  Running the evolutionary loop, which involves selection, mating, and
        mutation to produce new generations.
    5.  Tracking the best-performing ensemble and implementing early stopping
        if performance stagnates.
    6.  Evaluating the final best ensemble on a hold-out validation set.
    7.  Logging all results, progress, and artifacts to disk.
    """

    global_params: global_parameters
    """An instance of the `global_parameters` class."""

    ml_grid_object: Any
    """The main experiment object, containing data splits and configurations."""

    verbose: int
    """The verbosity level, inherited from `global_params`."""

    error_raise: bool
    """A flag to determine if errors should be raised, from `global_params`."""

    nb_params: List[int]
    """A list of possible values for the number of base learners in an ensemble."""

    pop_params: List[int]
    """A list of possible values for the population size."""

    g_params: List[int]
    """A list of possible values for the number of generations."""

    log_folder_path: str
    """The path to the directory for storing logs and artifacts."""

    creator: Any
    """The DEAP creator object for defining fitness and individuals."""

    toolbox: base.Toolbox
    """The DEAP toolbox containing the genetic operators."""

    project_score_save_object: project_score_save_class
    """An object for saving final scores to the master log file."""

    local_param_dict: Dict
    """A dictionary of local parameters for the current run."""

    def __init__(
        self,
        ml_grid_object: Any,
        local_param_dict: Dict,
        global_params: global_parameters,
    ):
        """Initializes the Genetic Algorithm runner.

        Args:
            ml_grid_object: The main experiment object, containing data splits
                and configurations.
            local_param_dict: A dictionary of local parameters for the current run.
            global_params: An initialized global_parameters object.
        """
        self.global_params = global_params

        self.ml_grid_object = ml_grid_object

        self.verbose = self.global_params.verbose

        self.error_raise = self.global_params.error_raise

        ga_grid = Grid(
            global_params=self.global_params, test_grid=ml_grid_object.testing
        )

        # pass in and get outside
        self.nb_params, self.pop_params, self.g_params = (
            ga_grid.nb_params,
            ga_grid.pop_params,
            ga_grid.g_params,
        )

        self.ml_grid_object = ml_grid_object

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        self.parameter_space_size = local_param_dict.get("param_space_size")

        self.log_folder_path = ml_grid_object.logging_paths_obj.log_folder_path

        # --- Explicitly create logging directories to prevent race conditions ---
        pathlib.Path(self.log_folder_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{self.log_folder_path}/progress_logs_scores/").mkdir(
            parents=True, exist_ok=True
        )

        # Save config and settings at GA runtime for logging
        self._save_runtime_config(local_param_dict)
        # --------------------------------------------------------------------

        self.global_param_str = self.ml_grid_object.logging_paths_obj.global_param_str

        self.additional_naming = self.ml_grid_object.logging_paths_obj.additional_naming

        self.gen_eval_score_threshold_early_stopping = (
            self.global_params.gen_eval_score_threshold_early_stopping
        )

        self.creator = creator

        self.tools = tools

        # Initialize DEAP creator (must be done before creating toolbox)
        if hasattr(creator, "FitnessMax"):
            delattr(creator, "FitnessMax")
        if hasattr(creator, "Individual"):
            delattr(creator, "Individual")

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Initialize toolbox
        self.toolbox = base.Toolbox()

        self.project_score_save_object = self.ml_grid_object.project_score_save_object

        # Store data references from ml_grid_object
        self.X_test = getattr(ml_grid_object, "X_test", None)
        self.y_test = getattr(ml_grid_object, "y_test", None)
        self.X_train = getattr(ml_grid_object, "X_train", None)
        self.y_train = getattr(ml_grid_object, "y_train", None)
        self.X_test_orig = getattr(ml_grid_object, "X_test_orig", None)
        self.y_test_orig = getattr(ml_grid_object, "y_test_orig", None)

        self.multiprocess = False
        self.local_param_dict = local_param_dict

    def _save_runtime_config(self, local_param_dict: Dict) -> None:
        """Saves a copy of the config and all configured settings to the results output folder.

        This method creates two files in the log folder:
        1. runtime_config.yml - A YAML file containing the user's configuration
        2. runtime_settings.json - A JSON file containing all runtime settings including local_param_dict

        Args:
            local_param_dict: A dictionary of local parameters for the current run.
        """
        import json

        try:
            # Save local parameters as JSON
            config_path = self.ml_grid_object.global_params.input_csv_path
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            runtime_settings = {
                "timestamp": timestamp,
                "config_file": (
                    config_path
                    if os.path.exists(config_path)
                    else "config.yml (default)"
                ),
                "local_parameters": local_param_dict,
                "global_parameters": {
                    "input_csv_path": self.global_params.input_csv_path,
                    "n_iter": self.global_params.n_iter,
                    # model_list is populated below with accurate names
                    "model_list": [],  # Placeholder, filled after lookup
                    "testing": self.global_params.testing,
                    "test_sample_n": self.global_params.test_sample_n,
                    "column_sample_n": self.global_params.column_sample_n,
                    "outcome_var_n": self.global_params.outcome_var_n,
                    "verbose": self.global_params.verbose,
                    "base_project_dir": self.global_params.base_project_dir,
                    "gen_eval_score_threshold_early_stopping": self.global_params.gen_eval_score_threshold_early_stopping,
                },
                "log_folder_path": self.log_folder_path,
                "param_space_index": getattr(
                    self.ml_grid_object, "param_space_index", "N/A"
                ),
            }

            # Build model list with actual names for accurate logging
            model_names = []
            for model in self.global_params.model_list:
                model_name = None
                if hasattr(model, "__name__"):
                    model_name = model.__name__
                elif isinstance(model, type):
                    model_name = model.__name__
                else:
                    # Try to find by reversing through MODEL_REGISTRY
                    for name, cls in self.global_params.MODEL_REGISTRY.items():
                        try:
                            if model is cls or (
                                hasattr(model, "__class__")
                                and model.__class__.__name__ == cls.__name__
                            ):
                                model_name = name
                                break
                        except Exception:
                            pass
                if model_name:
                    model_names.append(model_name)
                else:
                    # Fallback to string representation
                    model_names.append(str(model))

            settings_path = os.path.join(
                self.log_folder_path, f"runtime_settings_{timestamp}.json"
            )
            with open(settings_path, "w") as f:
                json.dump(runtime_settings, f, indent=2)

            # Also save a copy of the raw config file if it exists
            if os.path.exists(config_path):
                shutil.copy2(
                    config_path,
                    os.path.join(self.log_folder_path, "runtime_config.yml"),
                )

            logger.info("Runtime configuration saved to: %s", self.log_folder_path)

        except Exception as e:
            logger.warning("Failed to save runtime configuration: %s", e)

    def execute(self) -> List[List]:
        """Executes the full genetic algorithm process for all GA parameter combinations.

        This method iterates through a grid of GA-specific hyperparameters
        (number of base learners, population size, number of generations). For
        each combination, it runs a complete evolutionary process.

        The evolutionary loop within each run consists of selection, crossover,
        and mutation over multiple generations. It tracks the best individual
        and stops early if performance does not improve. Finally, it logs the
        results of the best ensemble found.

        Returns:
            A list of errors encountered during the execution. Each item in the
            list contains the model implementation, the exception, and a traceback.
        """
        logger.info("Executing GA runs...")
        self.model_error_list = []
        additional_naming = self.ml_grid_object.logging_paths_obj.additional_naming
        local_param_dict = self.ml_grid_object.local_param_dict

        grid = [self.nb_params, self.pop_params, self.g_params]

        param_grid = list(itertools.product(*grid))
        # date = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        # log_folder_path = f'log_{date}.txt'

        idx_list = [x for x in range(0, len(param_grid))]

        for i in enumerate(idx_list):
            i = i[0]
            param_grid_idx = i

            try:
                nb_val = param_grid[i][0]
                pop_val = param_grid[i][1]
                g_val = param_grid[i][2]

                # Only clear for first few runs or in short experiments
                if param_grid.index(param_grid[i]) < 3:
                    try:
                        clear_output(wait=True)
                    except Exception as e:
                        logger.warning("failed to clear output before run: %s", e)

                logger.info(
                    "Evolving ensemble: nb_val: %s, pop_val: %s, g_val: %s, ...",
                    nb_val,
                    pop_val,
                    g_val,
                )

                generation_progress_list = []

                start = time.time()

                logger.info("Registering toolbox elements")
                self.toolbox.register(
                    "ensembleGenerator",
                    ensembleGenerator,
                    nb_val=nb_val,
                    ml_grid_object=self.ml_grid_object,
                )
                self.toolbox.register(
                    "individual",
                    self.tools.initRepeat,
                    self.creator.Individual,
                    self.toolbox.ensembleGenerator,
                    n=1,  # could potentially increase this to pass to multiprocessing?
                    # ml_grid_object=self.ml_grid_object,
                )
                self.toolbox.register(
                    "population", self.tools.initRepeat, list, self.toolbox.individual
                )

                self.toolbox.register(
                    "evaluate",
                    evaluate_weighted_ensemble_auc,
                    ml_grid_object=self.ml_grid_object,
                )

                cx_type = local_param_dict.get("cx_type", "twopoint")

                if cx_type == "twopoint":
                    self.toolbox.register("mate", self.tools.cxTwoPoint)
                elif cx_type == "onepoint":
                    self.toolbox.register("mate", cxOnePoint)
                elif cx_type == "uniform":
                    uniform_indpb = local_param_dict.get("indpb", 0.5)
                    self.toolbox.register("mate", cxUniform, indpb=uniform_indpb)
                elif cx_type == "blend":
                    self.toolbox.register("mate", cxBlend)
                elif cx_type == "ordered":
                    self.toolbox.register("mate", cxOrdered)
                else:
                    logger.warning(
                        f"Unknown crossover type '{cx_type}', defaulting to 'twopoint'"
                    )
                    self.toolbox.register("mate", self.tools.cxTwoPoint)

                if cx_type == "uniform":
                    uniform_indpb = local_param_dict.get("indpb", 0.5)
                    self.toolbox.register(
                        "mutate", self.tools.mutFlipBit, indpb=uniform_indpb
                    )
                else:
                    self.toolbox.register(
                        "mutate",
                        self.tools.mutFlipBit,
                        indpb=local_param_dict.get("indpb", 0.05),
                    )
                self.toolbox.register("mutateFunction", mutateEnsemble)
                self.toolbox.register("mutateEnsemble", self.toolbox.mutateFunction)
                self.toolbox.register(
                    "select",
                    self.tools.selTournament,
                    tournsize=local_param_dict.get("t_size", 3),
                )

                start = time.time()

                if self.ml_grid_object.verbose >= 11:
                    logger.debug("self.toolbox.population pre evaluate: %s", pop_val)
                    logger.debug(self.toolbox.population)

                logger.info("Generate intial population n==%s", pop_val)
                pop = self.toolbox.population(n=pop_val)

                if self.ml_grid_object.verbose >= 11:
                    logger.debug("toolbox pre evaluate")
                    logger.debug(self.toolbox)
                    logger.debug(self.toolbox.evaluate)
                    logger.debug(pop)

                # Evaluate the entire population
                fitnesses = list(self.toolbox.map(self.toolbox.evaluate, pop))
                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values = fit

                # CXPB  is the probability with which two individuals
                #       are crossed
                #
                # MUTPB is the probability for mutating an individual
                CXPB, MUTPB = local_param_dict.get("cxpb"), local_param_dict.get(
                    "mutpb"
                )

                # Extracting all the fitnesses of
                fits = [ind.fitness.values[0] for ind in pop]

                if self.ml_grid_object.verbose >= 5:
                    mean_fit = sum(fits) / len(fits)
                    logger.info(
                        f"Initial pop: size={len(pop)}, mean_fitness={mean_fit:.4f}"
                    )

                # Log top 3 individuals from initial population (if verbose)
                if self.ml_grid_object.verbose >= 10:
                    sorted_fits = sorted(
                        enumerate(fits), key=lambda x: x[1], reverse=True
                    )[:3]
                    logger.debug("Top 3 initial individuals:")
                    for idx, fit in sorted_fits:
                        logger.debug(
                            f"  Ind {idx}: fitness={fit:.4f}, len_ensemble={len(pop[idx][0])}"
                        )

                # Variable keeping track of the number of generations
                g = 0

                # Begin the evolution
                y_test = self.ml_grid_object.y_test
                y_test_orig = self.ml_grid_object.y_test_orig

                chance_dummy_best_pred = [x for x in range(0, len(y_test))]

                try:
                    gen_eval_score = metrics.roc_auc_score(
                        y_test, chance_dummy_best_pred
                    )
                except ValueError:
                    gen_eval_score = 0.5
                gen_eval_score_counter = 0

                pbar = tqdm.tqdm(total=g_val + 1)
                # while currentData[0] <= runs:

                stop_early = False

                gen_eval_score_previous = gen_eval_score
                gen_eval_score_gain = 0

                highest_scoring_ensemble = (0, None)

                while g < g_val and gen_eval_score < 0.999 and not stop_early:
                    if self.global_params.progress_bars:
                        pbar = tqdm.tqdm(total=g_val + 1)

                    if self.ml_grid_object.verbose < 9:
                        # Only clear output every 5 generations for long runs to prevent GUI lag
                        if g <= 10 or (g + 1) % 5 == 0:
                            clear_output(wait=False)
                    # while g < 50: alt ::  while g < g_val and  ?? eval some how measure AUC or mcc of ensemble?
                    # for i in tqdm(range(0, g_val)):
                    # A new generation
                    g = g + 1
                    if self.global_params.progress_bars:
                        pbar.update(1)
                    # Log generation start (less verbose)
                    if self.ml_grid_object.verbose >= 1:
                        logger.info("--- Generation %d starting ---", g)

                    # Select the next generation individuals
                    offspring = self.toolbox.select(pop, len(pop))

                    # Clone the selected individuals
                    offspring = list(self.toolbox.map(self.toolbox.clone, offspring))

                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            self.toolbox.mate(child1[0], child2[0])
                            del child1.fitness.values
                            del child2.fitness.values

                    # Mutation counter
                    mut_count = 0
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            mutatedEnsemble = mutateEnsemble(
                                offspring[mut_count], ml_grid_object=self.ml_grid_object
                            )
                            offspring[mut_count] = mutatedEnsemble
                            del mutant.fitness.values
                        mut_count += 1

                    # Evaluate only individuals with invalid fitness
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit

                    pop[:] = offspring

                    # Gather all the fitnesses
                    fits = [ind.fitness.values[0] for ind in pop]
                    length = len(pop)
                    mean = sum(fits) / length
                    sum2 = sum(x * x for x in fits)
                    std = abs(sum2 / length - mean**2) ** 0.5

                    # Log generation stats (only every 5 generations unless verbose > 5)
                    if self.ml_grid_object.verbose >= 5 or g <= 5 or g % 5 == 0:
                        logger.info(
                            f"Gen {g}: pop_size={length}, min={min(fits):.4f}, max={max(fits):.4f}, "
                            f"mean={mean:.4f}, std={std:.4f}"
                        )
                    # pool.close() # experimental
                    # Additional eval stage for generation truncation:
                    # argmin... or argmax for auc

                    # Calculate the best individual from within the population
                    pop_size = len(pop)

                    # Find top performers without re-evaluating
                    fitness_list = [
                        (i, pop[i].fitness.values[0]) for i in range(pop_size)
                    ]
                    sorted_by_fitness = sorted(
                        fitness_list, key=lambda x: x[1], reverse=True
                    )

                    # Evaluate best individual more thoroughly
                    best_idx = sorted_by_fitness[0][0]
                    best = pop[best_idx]
                    # best_pred = get_best_y_pred(best)
                    # gen_eval_score = metrics.roc_auc_score(self.y_test_orig, best_pred)

                    # With best individual from population, evaluate their ensemble score on metric
                    y_pred = get_y_pred_resolver(
                        best, ml_grid_object=self.ml_grid_object, valid=False
                    )

                    try:
                        gen_eval_score = metrics.roc_auc_score(y_test, y_pred)
                        gen_mcc = metrics.matthews_corrcoef(y_test, y_pred)
                        gen_f1 = metrics.f1_score(y_test, y_pred, average="binary")
                        gen_accuracy = metrics.accuracy_score(y_test, y_pred)

                        # Measure diversity of best individual
                        try:
                            best_diversity = measure_binary_vector_diversity(best)
                        except Exception:
                            best_diversity = 0.0
                    except ValueError:
                        gen_eval_score = (
                            0.5  # Assign random chance score if AUC is not defined
                        )
                        gen_mcc = 0.0
                        gen_f1 = 0.0
                        gen_accuracy = 0.0
                        best_diversity = 0.0

                    # Log generation progress (less frequently unless verbose)
                    log_freq = 5 if self.ml_grid_object.verbose >= 5 else 10
                    if g <= 3 or g % log_freq == 0:
                        logger.info(
                            f"Gen {g}: Best AUC {gen_eval_score:.4f}, MCC {gen_mcc:.4f}, "
                            f"F1 {gen_f1:.4f}, Acc {gen_accuracy:.4f}, Div {best_diversity:.3f}"
                        )
                    generation_progress_list.append(gen_eval_score)

                    if gen_eval_score < highest_scoring_ensemble[0]:
                        gen_eval_score_counter = gen_eval_score_counter + 1
                        if self.verbose >= 1:
                            logger.info(
                                "gen_eval_score_counter %s, highest so far: %s",
                                gen_eval_score_counter,
                                highest_scoring_ensemble[0],
                            )

                        if (
                            gen_eval_score_counter
                            > self.gen_eval_score_threshold_early_stopping
                        ):
                            stop_early = True
                    elif gen_eval_score > highest_scoring_ensemble[0]:
                        if self.verbose >= 1:
                            logger.info(
                                "gen_eval_score gain: %s rate: %s ETA: %s",
                                gen_eval_score - gen_eval_score_previous,
                                (highest_scoring_ensemble[0] - 0.5) / g,
                                round(
                                    (
                                        (1 - highest_scoring_ensemble[0])
                                        / (gen_eval_score_gain + 1.00000000e-99)
                                    )
                                ),
                            )
                        gen_eval_score_gain = gen_eval_score_gain + (
                            gen_eval_score - gen_eval_score_previous
                        )
                        gen_eval_score_counter = 0

                    if gen_eval_score > highest_scoring_ensemble[0]:
                        highest_scoring_ensemble = (gen_eval_score, best)

                    gen_eval_score_previous = gen_eval_score

                if self.global_params.progress_bars:
                    pbar.close()

                # best = pop[np.argmax([toolbox.evaluate(x) for x in pop])] #was argmin

                # Get stored highest ensemble
                best = highest_scoring_ensemble[1]
                if self.verbose >= 1:
                    logger.info("\n")
                    logger.info("Best Ensemble Model: ")
                    for i in range(0, len(best[0])):
                        logger.info(
                            "%s n features: %s", best[0][i][1], len(best[0][i][2])
                        )

                end = time.time()
                if self.verbose >= 1:
                    logger.info(end - start)

                try:
                    if self.verbose >= 1:
                        logger.info(
                            "Getting final final best pred for plot with validation set, get weights from xtrain ytrain"
                        )
                    best_pred_orig = get_y_pred_resolver(
                        ensemble=best, ml_grid_object=self.ml_grid_object, valid=True
                    )
                    if self.verbose >= 1:
                        run_index = param_grid_idx
                        plot_basename = (
                            "best_pop="
                            + str(pop_val)
                            + "_g="
                            + str(g_val)
                            + "_nb="
                            + str(nb_val)
                        )
                        if run_index % 10 == 0:
                            plot_auc(
                                y_test_orig,
                                best_pred_orig,
                                plot_basename,
                            )
                        else:
                            fig = plt.figure()
                            plot_auc_base(
                                y_test_orig,
                                best_pred_orig,
                                plot_basename,
                                fig=fig,
                            )
                            plt.close(fig)
                        logger.info(
                            "nb_val: %s, pop_val: %s, g_val: %s", nb_val, pop_val, g_val
                        )
                        try:
                            final_auc = metrics.roc_auc_score(
                                y_test_orig, best_pred_orig
                            )
                            logger.info("AUC: %s, g: %s", final_auc, g)
                        except ValueError:
                            logger.warning(
                                "AUC: undefined (only one class in y_true), g: %s", g
                            )
                        # Calculate additional metrics for better visibility
                        try:
                            final_mcc = metrics.matthews_corrcoef(
                                y_test_orig, best_pred_orig
                            )
                            final_f1 = metrics.f1_score(
                                y_test_orig, best_pred_orig, average="binary"
                            )
                            final_precision = metrics.precision_score(
                                y_test_orig, best_pred_orig, average="binary"
                            )
                            final_recall = metrics.recall_score(
                                y_test_orig, best_pred_orig, average="binary"
                            )
                            final_accuracy = metrics.accuracy_score(
                                y_test_orig, best_pred_orig
                            )

                            # Measure diversity
                            diversity_metric = measure_binary_vector_diversity(best)

                            logger.info(
                                f"Best Ensemble: AUC {final_auc:.4f}, MCC {final_mcc:.4f}, F1 {final_f1:.4f}, "
                                f"Precision {final_precision:.4f}, Recall {final_recall:.4f}, Accuracy {final_accuracy:.4f}, "
                                f"diversity_score: {diversity_metric:.4f}"
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.error("Failed to get best y pred and plot auc")
                    logger.error(e)
                    logger.error("best_pred_orig fail:")

                    raise
                    pass

                end = time.time()

                # with open(self.global_param_str+self.additional_naming+"/progress_logs/"+log_folder_path, "a") as myfile:
                #     myfile.write(' '.join([str(i) for i in ["nb_val:", nb_val, "pop_val:", pop_val, "g_val:", g_val,
                #           "AUC: ", metrics.roc_auc_score(best_pred_orig, best_pred_orig), "Run Time (min): ", round((end - start)/60, 3), "g:", g]]))
                #     myfile.write('\n')
                #     myfile.close()

                try:
                    metrics.roc_auc_score(y_test_orig, best_pred_orig)
                except ValueError:
                    pass
                current_algorithm = best
                method_name = str(best)
                pg = "nan"
                n_iter_v = "nan"
                self.ml_grid_object.nb_val = nb_val
                self.ml_grid_object.pop_val = pop_val
                self.ml_grid_object.g_val = g_val
                self.ml_grid_object.g = g

                # Convert model definition to string for low file size
                best_str = best.copy()

                best_str_converted = best.copy()

                original_features = self.ml_grid_object.original_feature_names

                for i in range(0, len(best_str[0])):
                    best_str[0][i] = list(best_str[0][i])
                    best_str[0][i][1] = str(best_str[0][i][1])

                    best_str_converted[0][i] = list(best_str_converted[0][i])
                    best_str_converted[0][i][1] = str(best_str_converted[0][i][1])

                    # Convert feature list to binary vector
                    current_features = best_str_converted[0][i][2]
                    binary_feature_vector = [
                        1 if f in current_features else 0 for f in original_features
                    ]
                    best_str_converted[0][i][2] = binary_feature_vector

                    best_str_converted[0][i] = tuple(best_str_converted[0][i])

                try:
                    if self.verbose >= 1:
                        logger.info("Writing grid perturbation to log")
                    # Is for valid or no? Pass valid and set orig or ytest...
                    # write line to best grid scores---------------------
                    self.project_score_save_object.update_score_log(
                        # self=self.project_score_save_object,
                        ml_grid_object=self.ml_grid_object,
                        best_pred_orig=best_pred_orig,
                        current_algorithm=current_algorithm,
                        method_name=method_name,
                        pg=pg,
                        start=start,
                        n_iter_v=n_iter_v,
                        valid=True,
                        generation_progress_list=generation_progress_list,
                        best_ensemble=best_str_converted,
                        original_feature_names=self.ml_grid_object.original_feature_names,
                    )

                except Exception as e:
                    logger.error(e)
                    logger.error("Failed to upgrade grid entry")
                    raise

                # Construct the base path for plots robustly
                plot_base_path = os.path.join(
                    self.ml_grid_object.base_project_dir,
                    self.global_param_str + (additional_naming or ""),
                )

                run_index = param_grid_idx
                if run_index % 10 == 0:
                    plot_generation_progress_fitness(
                        generation_progress_list,
                        pop_val,
                        g_val,
                        nb_val,
                        file_path=plot_base_path,
                    )
                else:
                    fig = plt.figure()
                    plot_generation_progress_fitness_wrapper(
                        generation_progress_list,
                        pop_val,
                        g_val,
                        nb_val,
                        file_path=plot_base_path,
                        fig=fig,
                    )
                    plt.close(fig)

                with open(
                    self.ml_grid_object.base_project_dir
                    + (self.global_param_str or "")
                    + (self.additional_naming or "")
                    + "best_pop="
                    + str(pop_val)
                    + "_g="
                    + str(g_val)
                    + "_nb="
                    + str(nb_val)
                    + ".pkl",
                    "wb",
                ) as file:
                    # dump best_str instead of best which contains actual model.
                    # Unnecessary since ensemble can be fitted in deployment
                    # Can model always be stored as string during operation of GA?
                    pickle.dump(best_str, file)

                # Try to reset DEAP for second run. Should be seperate:
                from deap import base, creator

                del self.toolbox
                gc.collect()

                if hasattr(creator, "FitnessMax"):
                    delattr(creator, "FitnessMax")
                if hasattr(creator, "Individual"):
                    delattr(creator, "Individual")

                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
                self.toolbox = base.Toolbox()
                self.toolbox.register(
                    "evaluate",
                    evaluate_weighted_ensemble_auc,
                    ml_grid_object=self.ml_grid_object,
                )

            except Exception as Argument:
                try:
                    str_to_write = os.path.join(
                        f"{self.ml_grid_object.base_project_dir+self.global_param_str + additional_naming}/logs/",
                        "logging.txt",
                    )

                    logger.info(str_to_write)
                except Exception as e:
                    logger.error(e)
                    logger.error("failed to get base dir str?")

                f = open(
                    str_to_write,
                    "a",
                )

                # writing in the file
                f.write(str(Argument))
                f.write(str(traceback.format_exc()))
                f.close()
                raise
                # continue

        # for line in (master_result_list):
        #     print(line)

        return self.model_error_list
