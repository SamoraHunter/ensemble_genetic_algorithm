import datetime
import gc
import pathlib
import itertools
from typing import Any, Dict, List
import os
import pickle
import random
import logging
import time
import traceback

import ml_grid
import numpy as np
import pandas as pd
import tqdm
from deap import algorithms, base, creator, tools
from IPython.display import clear_output

# from ml_grid.ga_functions.ga_plots.ga_progress import plot_generation_progress_fitness
from ml_grid.ga_functions.ga_plots.ga_progress import plot_generation_progress_fitness
from ml_grid.model_classes_ga.logistic_regression_model import (
    logisticRegressionModelGenerator,
)
from ml_grid.pipeline.ensemble_generator_ga import ensembleGenerator
from ml_grid.pipeline.evaluate_methods_ga import (
    evaluate_weighted_ensemble_auc,
    get_y_pred_resolver,
    measure_binary_vector_diversity,
)
from ml_grid.pipeline.mutate_methods import mutateEnsemble
from ml_grid.pipeline.plot_methods.plot_auc_ga import plot_auc

# from ml_grid.model_classes import LogisticRegression_class
# from ml_grid.pipeline import grid_search_cross_validate
from ml_grid.util import grid_param_space
from ml_grid.util.global_params import global_parameters
from ml_grid.util.grid_param_space_ga import Grid
from ml_grid.util.project_score_save import project_score_save_class
from sklearn import metrics
from sklearn.model_selection import ParameterGrid

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

    def __init__(self, ml_grid_object: Any, local_param_dict: Dict):
        """Initializes the Genetic Algorithm runner.

        Args:
            ml_grid_object: The main experiment object, containing data splits
                and configurations.
            local_param_dict: A dictionary of local parameters for the current run.
        """
        self.global_params = global_parameters()

        self.ml_grid_object = ml_grid_object

        self.verbose = self.global_params.verbose

        self.error_raise = self.global_params.error_raise

        ga_grid = Grid(test_grid=ml_grid_object.testing)

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
        # --------------------------------------------------------------------

        self.global_param_str = self.ml_grid_object.logging_paths_obj.global_param_str

        self.additional_naming = self.ml_grid_object.logging_paths_obj.additional_naming

        self.gen_eval_score_threshold_early_stopping = (
            self.global_params.gen_eval_score_threshold_early_stopping
        )

        self.creator = creator

        self.tools = tools

        self.toolbox = base.Toolbox()

        self.project_score_save_object = self.ml_grid_object.project_score_save_object

        self.X_test = self.ml_grid_object.X_test
        self.y_test = self.ml_grid_object.y_test
        self.X_train = self.ml_grid_object.X_train
        self.y_train = self.ml_grid_object.y_train

        # self.X_train_orig = self.ml_grid_object.X_train_orig
        # self.y_train_orig = self.ml_grid_object.y_train_orig
        self.X_test_orig = self.ml_grid_object.X_test_orig
        self.y_test_orig = self.ml_grid_object.y_test_orig

        gen_eval_score_threshold_early_stopping = 5

        if __name__ == "__main__":
            grid = [self.nb_params, self.pop_params, self.g_params]  # type: ignore
            param_grid = list(itertools.product(*grid))
            logger.debug(param_grid)
            for elem in param_grid:
                logger.debug("%s %s model generation space", elem, elem[0] * elem[1])
                logger.debug("%s %s individual evaluation space", elem, elem[0] * elem[2])
            logger.debug(len(param_grid))

            prediction_array = None

            date = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            log_folder_path = f"log_{date}.txt"

            if self.verbose >= 2:
                logger.info(f"{len(self.model_class_list)} models loaded")

        self.multiprocess = False

        self.local_param_dict = local_param_dict

        # test
        self.creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        self.creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "evaluate", evaluate_weighted_ensemble_auc, ml_grid_object=ml_grid_object
        )

        if self.verbose >= 2:
            logger.info("Passed main GA init")

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

        global_param_str = self.ml_grid_object.logging_paths_obj.global_param_str

        additional_naming = self.ml_grid_object.logging_paths_obj.additional_naming

        global_param_dict = self.ml_grid_object.global_params

        log_folder_path = self.ml_grid_object.logging_paths_obj.log_folder_path

        local_param_dict = self.ml_grid_object.local_param_dict

        grid = [self.nb_params, self.pop_params, self.g_params]

        param_grid = list(itertools.product(*grid))
        logger.debug(param_grid)
        logger.debug(len(param_grid))

        prediction_array = None

        # date = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        # log_folder_path = f'log_{date}.txt'

        idx_list = [x for x in range(0, len(param_grid))]

        for i in enumerate(idx_list):
            i = i[0]

            try:
                nb_val = param_grid[i][0]
                pop_val = param_grid[i][1]
                g_val = param_grid[i][2]

                try:
                    clear_output(wait=True)
                except Exception as e:
                    logger.warning("failed to clear output before run: %s", e)

                logger.info(
                    "Evolving ensemble: nb_val: %s, pop_val: %s, g_val: %s, ...", nb_val, pop_val, g_val
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

                self.toolbox.register("mate", self.tools.cxTwoPoint)
                self.toolbox.register(
                    "mutate", self.tools.mutFlipBit, indpb=local_param_dict.get("indpb")
                )
                self.toolbox.register("mutateFunction", mutateEnsemble)
                self.toolbox.register("mutateEnsemble", self.toolbox.mutateFunction)
                self.toolbox.register(
                    "select",
                    self.tools.selTournament,
                    tournsize=local_param_dict.get("t_size"),
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

                if self.ml_grid_object.verbose >= 11:
                    logger.debug("fits")
                    logger.debug(fits)

                # Variable keeping track of the number of generations
                g = 0

                # Begin the evolution
                chance_dummy_best_pred = [x for x in range(0, len(self.y_test))]

                try:
                    gen_eval_score = metrics.roc_auc_score(
                        self.y_test, chance_dummy_best_pred
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

                while g < g_val and gen_eval_score < 0.999 and stop_early == False:

                    if self.ml_grid_object.verbose < 9 and g % 2 == 0:
                        clear_output(wait=False)
                    # while g < 50: alt ::  while g < g_val and  ?? eval some how measure AUC or mcc of ensemble?
                    # for i in tqdm(range(0, g_val)):
                    # A new generation
                    g = g + 1
                    pbar.update(1)
                    logger.info("\n -- Generation %i --", g)
                    # Select the next generation individuals
                    logger.info("Selecting next generation individuals, %s", len(pop))
                    offspring = self.toolbox.select(pop, len(pop))
                    # Clone the selected individuals
                    logger.info("Clone the selected individuals")
                    offspring = list(self.toolbox.map(self.toolbox.clone, offspring))
                    logger.info("Apply crossover and mutation on the offspring")
                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            self.toolbox.mate(child1[0], child2[0])
                            del child1.fitness.values
                            del child2.fitness.values
                    counter = 0
                    logger.info("mutate")
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            # print(mutant[0][0])
                            # mutant[0][0] = baseLearnerGenerator()#
                            # print(offspring[counter])
                            mutatedEnsemble = mutateEnsemble(
                                offspring[counter], ml_grid_object=self.ml_grid_object
                            )
                            offspring[counter] = mutatedEnsemble
                            # print("mutated into:")
                            # print(mutatedEnsemble)
                            # toolbox.mutateEnsemble(mutant[0])
                            # toolbox.mutate(mutant[0][0])
                            del mutant.fitness.values
                        counter = counter + 1
                    logger.info("Evaluate the individuals with an invalid fitness")
                    # Evaluate the individuals with an invalid fitness
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    # fitnesses = map(toolbox.evaluate, invalid_ind)
                    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
                    pop[:] = offspring
                    logger.info("Gather all the fitnesses in one list and print the stats")
                    # Gather all the fitnesses in one list and print the stats
                    fits = [ind.fitness.values[0] for ind in pop]
                    length = len(pop)
                    mean = sum(fits) / length
                    sum2 = sum(x * x for x in fits)
                    std = abs(sum2 / length - mean**2) ** 0.5
                    logger.info(
                        f"min: {min(fits)}, max: {max(fits)} , mean: {mean}, std: {std}"
                    )
                    # pool.close() # experimental
                    # Additional eval stage for generation truncation:
                    # argmin... or argmax for auc

                    # calculate the best individual from within the population by arg min or max on target metric
                    best = pop[np.argmax([self.toolbox.evaluate(x) for x in pop])]
                    # best_pred = get_best_y_pred(best)
                    # gen_eval_score = metrics.roc_auc_score(self.y_test_orig, best_pred)

                    # With best individual from population, evaluate their ensemble score on metric
                    y_pred = get_y_pred_resolver(
                        best, ml_grid_object=self.ml_grid_object, valid=False
                    )

                    try:
                        gen_eval_score = metrics.roc_auc_score(self.y_test, y_pred)
                    except ValueError:
                        gen_eval_score = 0.5  # Assign random chance score if AUC is not defined
                    logger.info("gen_eval_score == %s Generation %s", gen_eval_score, g)
                    generation_progress_list.append(gen_eval_score)

                    if gen_eval_score < highest_scoring_ensemble[0]:
                        gen_eval_score_counter = gen_eval_score_counter + 1
                        if self.verbose >= 1:
                            logger.info(
                                "gen_eval_score_counter %s, highest so far: %s", gen_eval_score_counter, highest_scoring_ensemble[0]
                            )

                        if (
                            gen_eval_score_counter
                            > self.gen_eval_score_threshold_early_stopping
                        ):
                            stop_early = True
                    elif gen_eval_score > highest_scoring_ensemble[0]:
                        if self.verbose >= 1:
                            logger.info(
                                "gen_eval_score gain: %s rate: %s ETA: %s", gen_eval_score-gen_eval_score_previous, (highest_scoring_ensemble[0]-0.5)/g, round(((1-highest_scoring_ensemble[0])/(gen_eval_score_gain+1.00000000e-99)))
                            )
                        gen_eval_score_gain = gen_eval_score_gain + (
                            gen_eval_score - gen_eval_score_previous
                        )
                        gen_eval_score_counter = 0

                    if gen_eval_score > highest_scoring_ensemble[0]:
                        highest_scoring_ensemble = (gen_eval_score, best)

                    gen_eval_score_previous = gen_eval_score

                pbar.close()

                # best = pop[np.argmax([toolbox.evaluate(x) for x in pop])] #was argmin

                # Get stored highest ensemble
                best = highest_scoring_ensemble[1]
                if self.verbose >= 1:
                    logger.info("\n")
                    logger.info("Best Ensemble Model: ")
                    for i in range(0, len(best[0])):
                        logger.info("%s n features: %s", best[0][i][1], len(best[0][i][2]))

                if self.verbose >= 1:
                    logger.info(
                        f"Best Ensemble diversity score: {measure_binary_vector_diversity(best)}"
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
                        plot_auc(
                            self.y_test_orig,
                            best_pred_orig,
                            "best_pop="
                            + str(pop_val)
                            + "_g="
                            + str(g_val)
                            + "_nb="
                            + str(nb_val),
                        )
                        logger.info("nb_val: %s, pop_val: %s, g_val: %s", nb_val, pop_val, g_val)
                        try:
                            final_auc = metrics.roc_auc_score(self.y_test_orig, best_pred_orig)
                            logger.info(
                                "AUC: %s, g: %s", final_auc, g
                            )
                        except ValueError:
                            logger.warning("AUC: undefined (only one class in y_true), g: %s", g)
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
                date = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

                # with open(f'{self.global_param_str+self.additional_naming}/results_master_lists/master_result_list_{date}.pkl', 'wb') as f:
                #     pickle.dump(master_result_list, f)

                try:
                    scores = metrics.roc_auc_score(self.y_test_orig, best_pred_orig)
                except ValueError:
                    scores = 0.5
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
                        scores=scores,
                        best_pred_orig=best_pred_orig,
                        current_algorithm=current_algorithm,
                        method_name=method_name,
                        pg=pg,
                        start=start,
                        n_iter_v=n_iter_v,
                        valid=True,
                        generation_progress_list=generation_progress_list,
                        best_ensemble=best_str_converted,
                    )

                except Exception as e:
                    logger.error(e)
                    logger.error("Failed to upgrade grid entry")
                    raise

                # Construct the base path for plots robustly
                plot_base_path = os.path.join(
                    self.ml_grid_object.base_project_dir,
                    self.global_param_str + additional_naming,
                )

                plot_generation_progress_fitness(
                    generation_progress_list,
                    pop_val,
                    g_val,
                    nb_val,
                    file_path=plot_base_path,
                )

                with open(
                    self.ml_grid_object.base_project_dir
                    + self.global_param_str
                    + self.additional_naming
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
                from deap import algorithms, base, creator, tools

                del self.toolbox
                gc.collect()
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
                self.toolbox = base.Toolbox()
                self.toolbox.register(
                    "evaluate",
                    evaluate_weighted_ensemble_auc,
                    ml_grid_object=self.ml_grid_object,
                )

            except Exception as Argument:
                date = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

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
