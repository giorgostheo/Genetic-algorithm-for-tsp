Genetic algorithm for the Traveling salesman problem

This is a simple but very effective genetic algorithm that uses techiques like crossover and mutation of a population to solve the tsp.

Using a predefined set of points and costs, it creates a population of size p.

By using both mutation and crossover, it ensures that the algorithm won't converge into a local minima.

The run_multiple_sims() and find_best_params() functions allow this script to be run multiple times for a set of hyperparameters and to be able to find the best set of hyperparameters for a scecified range.
