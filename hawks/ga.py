"""
Handles everything related to the GA itself. This is mainly setting up DEAP, processing the GA-specific arguments, and defining all the relevant aspects of the evolution (such as the parental selection, environmental selection etc.).
"""
from functools import partial
import operator

import numpy as np
from deap import base, creator, tools

from hawks.genotype import Genotype
from hawks.cluster import Cluster

def main_setup(objective_dict, dataset_obj, ga_params, constraint_params):
    # Create the DEAP toolbox
    toolbox = deap_setup(objective_dict, dataset_obj, ga_params)
    # Create the initial population, setup objectives, constraints etc.
    pop = initialize_ga(toolbox, ga_params, objective_dict, constraint_params)
    return toolbox, pop

def deap_setup(objective_dict, dataset_obj, ga_params):
    # Set up the weights for the optimisation
    weights = []
    for objective in objective_dict.values():
        weights.append(objective['class'].weight)
    # DEAP expects a tuple
    weights = tuple(weights)
    # Create the DEAP fitness class
    creator.create("Fitness", base.Fitness, weights=weights)  
    # Inherit from Genotype to get the methods
    creator.create("Individual", Genotype, fitness=creator.Fitness)
    # Create the DEAP toolbox (just being explicit here)
    toolbox = create_toolbox(objective_dict, dataset_obj, ga_params)
    return toolbox

def create_toolbox(objective_dict, dataset_obj, ga_params):
    # Instantiate toolbox
    toolbox = base.Toolbox()
    # Register the individual
    toolbox.register(
        "individual", generate_indiv,
        creator.Individual, dataset_obj=dataset_obj
    )
    # Register the population
    toolbox.register(
        "population", tools.initRepeat, list, toolbox.individual)
    # Select the mutation function
    toolbox = select_mutation(toolbox, dataset_obj, ga_params)
    # Select the crossover function
    toolbox = select_crossover(toolbox, ga_params)
    # Register the evaluation function
    toolbox.register("evaluate", evaluate_indiv, objective_dict=objective_dict)
    # Register parental selection
    toolbox.register(
        "parent_selection", parental_selection,
        offspring_size=ga_params["num_indivs"]
    )
    # Register environmental selection
    toolbox.register(
        "environment_selection", stochastic_ranking,
        ga_params=ga_params
    )
    return toolbox

def select_mutation(toolbox, dataset_obj, ga_params):
    # Select the appropriate mean mutation function
    if ga_params["mut_method_mean"] == "random":
        mut_mean_func = Cluster.mutate_mean_random
    elif ga_params["mut_method_mean"] == "pso":
        # **TODO**
        raise NotImplementedError
    else:
        raise ValueError(f"{ga_params['mut_method_mean']} is not a valid method to mutate the mean")
    # Create a partial function with the mean function and it's given arguments
    mut_mean_func = partial(mut_mean_func, **ga_params["mut_args_mean"])
    # Select the appropriate covariance mutation function
    if ga_params["mut_method_cov"] == "haar":
        mut_cov_func = Cluster.mutate_cov_haar
    else:
        raise ValueError(f"{ga_params['mut_method_cov']} is not a valid method to mutate the covariance")
    # Create a partial function with the covariance function and it's given arguments
    mut_cov_func = partial(mut_cov_func, **ga_params["mut_args_cov"])
    # Register the mutation function
    toolbox.register(
        "mutate",
        Genotype.mutation,
        mut_mean_func=mut_mean_func,
        mut_cov_func=mut_cov_func
    )
    # Check the mean probability
    if ga_params["mut_prob_mean"] == "length":
        mutpb_mean = 1/dataset_obj.num_clusters
    elif isinstance(ga_params["mut_prob_mean"], float):
        mutpb_mean = ga_params["mut_prob_mean"]
    else:
        raise TypeError(f"Could not understand mean mutation probability ({ga_params['mut_prob_mean']} of type {type(ga_params['mut_prob_mean'])})")
    # Set the mean probability
    Genotype.mutpb_mean = mutpb_mean
    # Check the covariance probability
    if ga_params["mut_prob_cov"] == "length":
        mutpb_cov = 1/dataset_obj.num_clusters
    elif isinstance(ga_params["mut_prob_cov"], float):
        mutpb_cov = ga_params["mut_prob_cov"]
    else:
        raise TypeError(f"Could not understand cov mutation probability ({ga_params['mut_prob_cov']} of type {type(ga_params['mut_prob_cov'])})")
    # Set the covariance probability
    Genotype.mutpb_cov = mutpb_cov
    return toolbox

def select_crossover(toolbox, ga_params):
    # Set up crossover/mate operator
    if ga_params["mate_scheme"] == "cluster":
        mate_func = Genotype.xover_cluster
    elif ga_params["mate_scheme"] == "dv":
        mate_func = Genotype.xover_genes
    else:
        raise ValueError(f'{ga_params["mate_scheme"]} is not a valid mutation scheme')
    # Register crossover
    toolbox.register(
        "mate", mate_func, cxpb=ga_params["mate_prob"])
    return toolbox

def generate_indiv(icls, dataset_obj):
    # Create the individual
    # Uses the DEAP wrapper around Genotype()
    indiv = icls([Cluster(size) for size in dataset_obj.cluster_sizes])
    # Create the views (each cluster.values is a view into genotype.all_values)
    indiv.create_views()
    # And sample some initial values
    indiv.resample_values()
    return indiv

def setup_objectives(objective_dict, pop):
    setup_funcs = []
    for objective in objective_dict.values():
        if hasattr(objective['class'], "setup_indiv"):
            setup_funcs.append(objective['class'].setup_indiv)
    for setup_func in setup_funcs:
        for indiv in pop:
            setup_func(indiv)

def initialize_ga(toolbox, ga_params, objective_dict, constraint_params):
    # Create the initial population
    pop = toolbox.population(n=ga_params['num_indivs'])
    # Setup what we need for the objectives (if anything)
    setup_objectives(objective_dict, pop)
    # Evaluate the initial population
    fitnesses = [toolbox.evaluate(indiv) for indiv in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # Calculate the constraints
    for indiv in pop:
        indiv.calc_constraints(constraint_params)
        reset_changed_flags(indiv)
    return pop

def evaluate_indiv(indiv, objective_dict):
    obj_values = []
    # Loop over the objectives
    for objective in objective_dict.values():
        # Calculate this objective
        res = objective['class'].eval_objective(indiv)#, **objective['args'])
        # Append the result
        obj_values.append(res)
    return tuple(obj_values)

def reset_changed_flags(indiv):
    # Set all flags to false to enable future partial recomp
    for cluster in indiv:
        cluster.changed = False

def parental_selection(pop, offspring_size):    
    parents = []
    # Just in case offspring size is different to popsize
    length = len(pop)
    # Create a list of tuples, each with 2 parents in
    for _ in range(int(offspring_size/2)):
        # Take the minimum as the pop is sorted in ranking order
        index1 = np.min(Genotype.global_rng.randint(length, size=2))
        index2 = np.min(Genotype.global_rng.randint(length, size=2))
        # Add this pair to the parent list
        parents.append(
            (pop[index1], pop[index2])
        )
    # Duplicate if we have a pop size of 1
    if not parents:
        parents = [(pop[0], pop[0])]
    return parents

def stochastic_ranking(pop, ga_params):
    # Avoid lookups for things in loop
    prob_fitness = ga_params["prob_fitness"]
    # Loop over each individual
    for _ in range(len(pop)):
        # Counter for break criteria
        swap_count = 0
        # Sweep over the population
        for j in range(len(pop)-1):
            # Get a random number
            u = Genotype.global_rng.rand()
            # Test if they're both feasible or if we eval based on objective
            if (pop[j].penalty == 0 and pop[j+1].penalty == 0) or (u < prob_fitness):
                # import pdb; pdb.set_trace()
                # Compare fitness of the two indivs
                if pop[j].fitness.wvalues < pop[j+1].fitness.wvalues:
                    # Swap the individuals
                    pop[j], pop[j+1] = pop[j+1], pop[j]
                    # Add to swap counter
                    swap_count += 1
            # Else test for penalty
            elif pop[j].penalty > pop[j+1].penalty:
                # Swap the individuals
                pop[j], pop[j+1] = pop[j+1], pop[j]
                # Add to swap counter
                swap_count += 1
        # Break when no swaps
        if swap_count == 0:
            break
    # Calculate how many of the best inidividuals to keep
    num_elites = int(ga_params["num_indivs"] * ga_params["elites"])
    # Return the appropriate number of individuals
    if len(pop) > num_elites:
        # Select the elites, then select the rest randomly from the remainder
        pop = pop[:num_elites] + [
            pop[num_elites:][i] for i in Genotype.global_rng.choice(
                len(pop[num_elites:]),
                size=ga_params["num_indivs"] - num_elites,
                replace=False
                )
            ]
    return pop

def generation(pop, toolbox, constraint_params):
    # Clone population for offspring
    offspring = [toolbox.clone(ind) for ind in pop]
    # Reconstruct the array views lost by cloning
    for indiv in offspring:
        indiv.recreate_views()
    # Select the parents
    offspring = toolbox.parent_selection(offspring)
    # Create offspring - this happens in place so the parents actually become the children (hence the cloning)
    for parent1, parent2 in offspring:
        # Crossover
        toolbox.mate(parent1, parent2)
        # Mutation
        toolbox.mutate(parent1)
        toolbox.mutate(parent2)
    # Flatten the list of tuples
    offspring = [parent for tup in offspring for parent in tup]
    # Resample the values
    for indiv in offspring:
        indiv.resample_values()
    # Evaluate the offspring
    fitnesses = [toolbox.evaluate(indiv) for indiv in offspring]
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit
    # Calculate the constraints (for changed clusters only)
    for indiv in offspring:
        indiv.recalc_constraints(constraint_params)
        # Reset every .changed flag to False
        reset_changed_flags(indiv)
    # Select from the current population and new offspring
    pop = toolbox.environment_selection(pop+offspring)
    return pop
