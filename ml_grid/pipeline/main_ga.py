import datetime
import gc
import itertools
import os
import pickle
import random
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
from sklearn import metrics
from sklearn.model_selection import ParameterGrid


class run:
    def __init__(self, ml_grid_object, local_param_dict):  # kwargs**
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
            grid = [self.nb_params, self.pop_params, self.g_params]
            param_grid = list(itertools.product(*grid))
            print(param_grid)
            for elem in param_grid:
                print(elem, elem[0] * elem[1], "model generation space")
                print(elem, elem[0] * elem[2], "individual evaluation space")
            print(len(param_grid))

            prediction_array = None

            date = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            log_folder_path = f"log_{date}.txt"

            if self.verbose >= 2:
                print(f"{len(self.model_class_list)} models loaded")

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
            print(f"Passed main GA init")

    def execute(self):
        print("Executing")
        self.model_error_list = []

        global_param_str = self.ml_grid_object.logging_paths_obj.global_param_str

        additional_naming = self.ml_grid_object.logging_paths_obj.additional_naming

        global_param_dict = self.ml_grid_object.global_params

        log_folder_path = self.ml_grid_object.logging_paths_obj.log_folder_path

        local_param_dict = self.ml_grid_object.local_param_dict

        grid = [self.nb_params, self.pop_params, self.g_params]

        param_grid = list(itertools.product(*grid))
        print(param_grid)
        for elem in param_grid:
            print(elem, elem[0] * elem[1], "model generation space")
            print(elem, elem[0] * elem[2], "individual evaluation space")
        print(len(param_grid))

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
                    print("failed to clear output before run", e)
                
                print(
                    "Evolving ensemble: ",
                    "nb_val:",
                    nb_val,
                    "pop_val:",
                    pop_val,
                    "g_val:",
                    g_val,
                    "...",
                )

                generation_progress_list = []

                start = time.time()

                print("Registering toolbox elements")
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
                    print("self.toolbox.population pre evaluate", pop_val)
                    print(self.toolbox.population)

                print(f"Generate intial population n=={pop_val}")
                pop = self.toolbox.population(n=pop_val)

                if self.ml_grid_object.verbose >= 11:
                    print("toolbox pre evaluate")
                    print(self.toolbox)
                    print(self.toolbox.evaluate)
                    print(pop)

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
                    print("fits")
                    print(fits)

                # Variable keeping track of the number of generations
                g = 0

                # Begin the evolution
                chance_dummy_best_pred = [x for x in range(0, len(self.y_test))]

                gen_eval_score = metrics.roc_auc_score(
                    self.y_test, chance_dummy_best_pred
                )

                gen_eval_score_counter = 0

                pbar = tqdm.tqdm(total=g_val + 1)
                # while currentData[0] <= runs:

                stop_early = False

                gen_eval_score_previous = gen_eval_score
                gen_eval_score_gain = 0

                highest_scoring_ensemble = (0, None)

                while g < g_val and gen_eval_score < 0.999 and stop_early == False:
                    
                    if(self.ml_grid_object.verbose<9):
                        clear_output(wait=False)
                    # while g < 50: alt ::  while g < g_val and  ?? eval some how measure AUC or mcc of ensemble?
                    # for i in tqdm(range(0, g_val)):
                    # A new generation
                    g = g + 1
                    pbar.update(1)
                    print("\n -- Generation %i --" % g)
                    # Select the next generation individuals
                    print("Selecting next generation individuals, ", len(pop))
                    offspring = self.toolbox.select(pop, len(pop))
                    # Clone the selected individuals
                    print("Clone the selected individuals")
                    offspring = list(self.toolbox.map(self.toolbox.clone, offspring))
                    print("Apply crossover and mutation on the offspring")
                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            self.toolbox.mate(child1[0], child2[0])
                            del child1.fitness.values
                            del child2.fitness.values
                    counter = 0
                    print("mutate")
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
                    print("Evaluate the individuals with an invalid fitness")
                    # Evaluate the individuals with an invalid fitness
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    # fitnesses = map(toolbox.evaluate, invalid_ind)
                    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
                    pop[:] = offspring
                    print("Gather all the fitnesses in one list and print the stats")
                    # Gather all the fitnesses in one list and print the stats
                    fits = [ind.fitness.values[0] for ind in pop]
                    length = len(pop)
                    mean = sum(fits) / length
                    sum2 = sum(x * x for x in fits)
                    std = abs(sum2 / length - mean**2) ** 0.5
                    print(
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

                    gen_eval_score = metrics.roc_auc_score(self.y_test, y_pred)
                    print(f"gen_eval_score == {gen_eval_score} Generation {g}")
                    generation_progress_list.append(gen_eval_score)

                    if gen_eval_score < highest_scoring_ensemble[0]:
                        gen_eval_score_counter = gen_eval_score_counter + 1
                        if self.verbose >= 1:
                            print(
                                f"gen_eval_score_counter {gen_eval_score_counter}, highest so far: {highest_scoring_ensemble[0]}"
                            )

                        if (
                            gen_eval_score_counter
                            > self.gen_eval_score_threshold_early_stopping
                        ):
                            stop_early = True
                    elif gen_eval_score > highest_scoring_ensemble[0]:
                        if self.verbose >= 1:
                            print(
                                f"gen_eval_score gain: {gen_eval_score-gen_eval_score_previous} rate: {(highest_scoring_ensemble[0]-0.5)/g} ETA: {round(((1-highest_scoring_ensemble[0])/(gen_eval_score_gain+1.00000000e-99)))}"
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

                print("\n")
                print("Best Ensemble Model: ")
                for i in range(0, len(best[0])):
                    print(best[0][i][1], "n features: ", len(best[0][i][2]))

                print(
                    f"Best Ensemble diversity score: {measure_binary_vector_diversity(best)}"
                )

                end = time.time()
                print(end - start)

                try:
                    print(
                        "Getting final final best pred for plot with validation set, get weights from xtrain ytrain"
                    )
                    best_pred_orig = get_y_pred_resolver(
                        ensemble=best, ml_grid_object=self.ml_grid_object, valid=True
                    )

                    plot_auc(
                        best_pred_orig,
                        "best_pop="
                        + str(pop_val)
                        + "_g="
                        + str(g_val)
                        + "_nb="
                        + str(nb_val),
                    )
                    print(
                        "nb_val:",
                        nb_val,
                        "pop_val:",
                        pop_val,
                        "g_val:",
                        g_val,
                        "AUC: ",
                        metrics.roc_auc_score(self.y_test_orig, best_pred_orig),
                        "g:",
                        g,
                    )
                except Exception as e:
                    print("Failed to get best y pred and plot auc")
                    print(e)
                    print("best_pred_orig fail:")

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

                scores = metrics.roc_auc_score(self.y_test_orig, best_pred_orig)
                current_algorithm = best
                method_name = str(best)
                pg = "nan"
                n_iter_v = "nan"

                try:
                    print("Writing grid perturbation to log")
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
                    )

                except Exception as e:
                    print(e)
                    print("Failed to upgrade grid entry")
                    raise

                # Convert model definition to string for low file size
                best_str = best.copy()

                for i in range(0, len(best_str[0])):
                    best_str[0][i] = list(best_str[0][i])
                    best_str[0][i][1] = str(best_str[0][i][1])
                    best_str[0][i] = tuple(best_str[0][i])

                plot_path = f"{self.ml_grid_object.base_project_dir+self.global_param_str + additional_naming}/"

                plot_generation_progress_fitness(
                    generation_progress_list,
                    pop_val,
                    g_val,
                    nb_val,
                    file_path=plot_path,
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

                    print(str_to_write)
                except Exception as e:
                    print(e)
                    print("failed to get base dir str?")

                f = open(
                    str_to_write,
                    "a",
                )

                # writing in the file
                f.write(str(Argument))
                f.write(str(traceback.format_exc()))
                # closing the file
                f.close()
                print(Argument)

                print(traceback.format_exc())
                raise
                # continue

        # for line in (master_result_list):
        #     print(line)

        return self.model_error_list
