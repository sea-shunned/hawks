"""
A few functions that tie multiple things together to do the setup. Could be integrated into the Generator class in the future.
"""
from hawks.genotype import Genotype
import hawks.objectives as objectives
import hawks.ga as ga

def setup_ga(ga_params, constraint_params, objective_params, dataset_obj):
    # Validate the constraints parameters
    Genotype.validate_constraints(constraint_params)
    # Setup the objective parameters
    objective_dict = _setup_objectives(objective_params)
    # Create the DEAP toolbox and generate the initial population
    toolbox, initial_pop = ga.main_setup(
        objective_dict, dataset_obj, ga_params, constraint_params
    )
    return objective_dict, toolbox, initial_pop

def _setup_objectives(objective_params):
    # Get the currently available/implemented objectives
    avail_objectives = {
        cls.__name__.lower():{'class':cls} for cls in objectives.ClusterIndex.__subclasses__()
    }
    # Create a dict to hold the objectives we select
    objective_dict = {}
    # Loop through the specified objectives
    for selected_obj in objective_params:
        selected_obj = selected_obj.lower()
        # Try to find it in the available objectives
        try:
            avail_objectives[selected_obj]
        # If we can't find, say it's not been implemented
        # More informative than a KeyError
        except KeyError:
            raise NotImplementedError(f"{selected_obj} is not implemented")
        # Create the key:value in our dict to pass on
        objective_dict[selected_obj] = avail_objectives[
            selected_obj]
        # Get the params for the objective(s)
        obj_args = objective_params[selected_obj]
        # Just for completeness
        objective_dict[selected_obj]["kwargs"] = obj_args
        # Set the kwargs for the class
        objective_dict[selected_obj]['class'].set_kwargs(obj_args)
    return objective_dict
