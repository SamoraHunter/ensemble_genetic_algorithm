# Genetic Algorithm Deep Dive

This guide provides a more detailed, code-oriented explanation of the core genetic algorithm (GA) process implemented in this project. It complements the high-level [Architectural Overview](docs_wiki/Architectural_Overview.md) and the theoretical [Technical Deep Dive](docs_wiki/Technical-Deep-Dive.md).

---

## The Core Evolutionary Loop

The genetic algorithm, implemented using the `DEAP` library, evolves a population of candidate solutions (ensembles) over several generations to find the one with the best predictive performance. Here’s a breakdown of the key concepts and steps.

### 1. Individual Representation (The Chromosome)

In this framework, an **individual** is an entire machine learning ensemble. Its "chromosome" is a Python list containing several **base learners**.

-   **Base Learner**: A base learner is a single, trained machine learning model (e.g., a specific `RandomForestClassifier` with its own tuned hyperparameters and trained on a specific subset of features).
-   **Ensemble (Individual)**: An individual is simply a collection of these base learners. For example: `[<XGBoost_model_1>, <LogisticRegression_model_A>, <RandomForest_model_7>]`.

The GA's goal is to find the optimal combination of these base learners.

### 2. Fitness Evaluation

The "fitness" of an individual (ensemble) is a measure of how well it performs its task.

1.  **Prediction**: The predictions from all base learners in the ensemble are collected.
2.  **Weighting**: A weighting method is applied to combine these predictions into a single ensemble prediction. This is controlled by the `ensemble_weighting_method` parameter:
    -   `'unweighted'`: The predictions are simply averaged.
    -   `'de'`: **Differential Evolution** is used to find a set of weights that maximizes the ensemble's AUC on the training data.
    -   `'ann'`: A small **Artificial Neural Network** is trained to learn the optimal, non-linear combination of base learner predictions.
3.  **Scoring**: The final combined prediction is evaluated against the validation set using a fitness metric. While the project logs multiple metrics like AUC, the primary fitness function for the GA's evolution is typically the **Matthews Correlation Coefficient (MCC)**, which is a robust metric for binary classification.

The resulting score is the individual's fitness. Higher is better.

### 3. Selection (Choosing Parents)

To create the next generation, the algorithm selects "parents" from the current population. Individuals with higher fitness are more likely to be chosen. This project uses **Tournament Selection**:

1.  A small, random subset of individuals is selected from the population (the size of this subset is the `tournament_size`).
2.  The individual with the best fitness within that small group is chosen as a parent.
3.  This process is repeated until enough parents are selected to create the next generation.

### 4. Crossover (Creating Offspring)

Crossover combines the genetic material of two parent ensembles to create one or more "offspring" ensembles. This project uses **Two-Point Crossover**:

1.  Two parent ensembles (lists of base learners) are chosen.
2.  Two random points are selected along the length of the chromosomes (the lists).
3.  The offspring is created by taking the segment of base learners from the first parent between the two points and filling the rest with base learners from the second parent.

This allows beneficial combinations of base learners from different successful ensembles to be combined. The probability of crossover occurring is controlled by the `crossover_rate` (`cxpb`).

### 5. Mutation (Introducing Variation)

Mutation introduces random changes into an individual's chromosome. This is crucial for maintaining genetic diversity and preventing the algorithm from getting stuck in a local optimum.

In this framework, mutation involves **swapping one base learner** in the ensemble with a brand new, randomly generated base learner from the initial pool.

The probability of an individual undergoing mutation is controlled by the `mutation_rate` (`mutpb`).

### 6. The Full Cycle

The GA process is as follows:

1.  **Initialization**: An initial population of random ensembles is created.
2.  **Evaluation**: The fitness of every individual in the population is calculated.
3.  **Loop (for `n_generations`)**:
    a. **Selection**: Parents are selected from the current population using tournament selection.
    b. **Crossover**: Offspring are created by applying crossover to pairs of parents.
    c. **Mutation**: Some offspring undergo mutation.
    d. **Replacement**: The old population is replaced by the new generation of offspring.
    e. **Evaluation**: The fitness of the new individuals is calculated.
4.  **Termination**: The loop stops after `n_generations` or if an early stopping criterion is met (e.g., no improvement in the best fitness for a certain number of generations).

The best individual found throughout this entire process is the final, optimized ensemble for that experiment run.