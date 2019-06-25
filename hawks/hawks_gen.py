"""
Defines the overarching Generator class for HAWKS. Intended to be the outward-facing class that users interact with, pulling everything else together.

Refactoring/abstraction may be required as more subclasses of the BaseGenerator are added.
"""
from itertools import product
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from copy import deepcopy
import json
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd

import hawks.ga as ga
import hawks.plotting as plotting
import hawks.prepare as prepare
import hawks.utils as utils
from hawks.cluster import Cluster
from hawks.dataset import Dataset
from hawks.genotype import Genotype

warnings.filterwarnings("ignore", module="deap", category=RuntimeWarning)

def create_generator(config=None):
    """Creates a generator instance according to the input parameters. If no argument is given, default values will be used according to "defaults.json". 
    
    Keyword Arguments:
        config {str/Path, or dict} -- Either a dict or a string/Path to the JSON that specifies the parameters to be given to the generator. Any omitted parameter will take from "defaults.json". (default: {None})
    
    Returns:
        BaseGenerator subclass -- An instance of the appropriate class of the BaseGenerator (i.e. SingleObjective) will be returned, according to input
    """
    # Load the default if nothing is given
    if config is None:
        warnings.warn(
            message=f"No config has been provided, so all default values will be used",
            category=UserWarning
        )
        config = BaseGenerator.load_default_config()
    # If a string (assumed a path) is given, process it
    elif isinstance(config, (str, Path)):
        param_path = Path(config)
        # Check if the given path actually exists
        if param_path.is_file():
            # Load the file and process the config
            config = utils.load_json(param_path)
        # Error if it doesn't
        else:
            raise FileNotFoundError(f"{config} was not found")
    # Skip straight to the processing if it's a dict
    elif isinstance(config, dict):
        pass
    else:
        raise TypeError(f"{type(config)} is not a valid type for the parameters")
    # Process the config
    config, any_saving = BaseGenerator.process_config(config)
    # Check if the config specify multiple sets of parameters or not
    multi_config = BaseGenerator._check_multiconfig(config)
    # Decide which instance we need from the config
    if config["hawks"]["mode"] == "single":
        hawks_obj = SingleObjective(config, any_saving, multi_config)
    else:
        raise ValueError(f"{config['hawks']['mode']} is not valid")
    return hawks_obj

class BaseGenerator:
    def __init__(self, config, any_saving, multi_config):
        # Keep the full config
        self.full_config = config
        # Flag for if any saving is occuring
        self.any_saving = any_saving
        # Flag for if the config specifies multiple sets or not
        self.multi_config = multi_config
        # Set the HAWKS parameters individually
        self.folder_name = config["hawks"]["folder_name"]
        self.mode = config["hawks"]["mode"]
        self.n_objectives = config["hawks"]["n_objectives"]
        self.num_runs = config["hawks"]["num_runs"]
        self.seed_num = config["hawks"]["seed_num"]
        self.save_best_data = config["hawks"]["save_best_data"]
        self.save_stats = config["hawks"]["save_stats"]
        self.plot_best = config["hawks"]["plot_best"]
        self.save_plot = config["hawks"]["save_plot"]
        # Initialize attributes
        self.population = None # Reference to the final pop
        self.best_dataset = None # Array of the best dataset
        self.best_indiv = [] # Best indiv Genotype object
        self.best_config = None # Config for the best indiv
        self.base_folder = None # Folder where everything is saved
        self.global_rng = None # RandomState instance used throughout
        self.config_list = None # Container for storing each of the configs
        self.stats = None # DataFrame of stats stored throughout the run
        # Create the folder if saving is required
        if self.any_saving:
            self.create_folder()

    @staticmethod
    def process_config(params):
        # Load the default config
        base_config = BaseGenerator.load_default_config()
        # Loop through the supplied parameters
        for first_key, value in params.items():
            # It should be a dict of subdicts, for hawks, dataset, objective etc.
            if isinstance(value, dict):
                # Loop through these items
                for second_key, param in value.items():
                    # If not value has provided, just keep the default
                    if param is not None:
                        # Try setting the parameter
                        try:
                            _ = base_config[first_key][second_key]
                            base_config[first_key][second_key] = param
                        # A typo or mistake has been made
                        except KeyError as e:
                            raise Exception(f"'{second_key}' is not a valid key for the config (in '{first_key}' settings)") from e
            # This should not be needed, but present in case
            else:
                if value is not None:
                    try:
                        # _ = base_config[first_key]
                        base_config[first_key] = value
                    except KeyError as e:
                        raise Exception(f"'{first_key}' is not a valid key for the config") from e
        # Check if there is a True in any of the saving parameters
        any_saving = any([v for k, v in base_config["hawks"].items() if "save" in k])
        # Check if a folder name is given, and if any saving is occurring
        if base_config["hawks"]["folder_name"] is None:
            # If there is saving, create a name from the time
            if any_saving:
                base_config["hawks"]["folder_name"] = datetime.today().strftime("%Y_%m_%d-%H%M%S")
        return base_config, any_saving

    def create_folder(self):
        """Create the folder(s) necessary for saving. This function is only called if any saving is specified.
        """
        # Set the path for the base folder
        self.base_folder = Path.cwd() / "hawks_experiments" / self.folder_name
        # Print a warning 
        if self.base_folder.is_dir():
            warnings.warn(
                message=f"{self.base_folder} already exists - previous results may be overwritten!",
                category=UserWarning
            )
        self.base_folder.mkdir(exist_ok=True, parents=True)
        # Make a datasets folder if needed
        if self.save_best_data:
            Path(self.base_folder / "datasets").mkdir(exist_ok=True)
        # Make a plots folder is need be
        if self.save_plot:
            Path(self.base_folder / "plots").mkdir(exist_ok=True)

    @staticmethod
    def load_default_config():
        default_config = utils.load_json(Path(__file__).parent / "defaults.json")
        return default_config

    def save_config(self, config, folder=None, filename=None):
        # Make sure that the config is complete
        # Ensure this works as expected (filling in any gaps)
        # Unlikely that this is needed as this method should be called after a run
        config, _ = BaseGenerator.process_config(config)
        # Determine location of the config
        # If no folder is given, default to the configs folder
        if folder is None:
            folder = Path.cwd()
        # Otherwise put it where specified
        # Some trust is in the user here
        else:
            folder = Path(folder)
            folder.mkdir(exist_ok=True, parents=True)
        # Create the path to the config
        if filename is None:
            fpath = folder / f"{self.folder_name}_config.json"
        # Use the filename if given
        else:
            fpath = folder / f"{filename}.json"
        # Save the config
        with open(fpath, "w") as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def _check_multiconfig(config):
        """Check if a list exists in the config i.e. it defines a set of parameters
        
        Arguments:
            config {dict} -- Config dictionary
        
        Returns:
            bool -- Whether the config defines a set of parameters
        """
        # Initialize flag
        found_list = False
        # Loop through the input
        for val in config.values():
            # If it's a dict, recurse
            if isinstance(val, dict):
                found_list = found_list or BaseGenerator._check_multiconfig(val)
            # Set flag to True if a list is found
            elif isinstance(val, list):
                found_list = True
            # Break early if possible
            if found_list:
                break
        return found_list

    def get_stats(self):
        """Return the stats (pandas DataFrame), useful in an interactive setting
        """
        return self.stats
    
    def save_stats_csv(self, fpath):
        """Save the stats DataFrame to the specified location
        """
        self.stats.to_csv(
            Path(fpath),
            index=False
        )

    def get_config(self):
        """Return the full config, useful in an interactive setting
        """
        return self.full_config

    def _get_configs(self, key_paths, param_lists):
        """
        I need a function here that, in the non-multi-config case, just returns one thing (self.full_config).

        In the multi_config case, I need it to iteratively return (i.e. yield) each config (where the rough steps needed are outlined below).

        Does the below work at all? Is there a better way?
        """
        # Check if the config specifies a set or not
        if self.multi_config:
            # Create all the possible combinations for these sets 
            for params in product(*param_lists):
                # Create a copy of the config
                config = deepcopy(self.full_config)
                # Set the specific params for this combination
                for key_path, param in zip(key_paths, params):
                    utils.set_key_path(config, key_path, param)
                # Yield the single set config
                yield params, config
        # Yield the only single set config
        else:
            yield None, self.full_config

    def _count_multiconfigs(self):
        # Get the key paths to all the parameter sets, and the sets themselves
        key_paths, param_lists = utils.get_key_paths(
            self.full_config
        )
        # Check if a reasonable number of configs have been configured
        try:
            total_configs = len(list(product(*param_lists)))
            if total_configs * self.num_runs > 1000:
                warnings.warn(
                    f"{total_configs} configs will be generated, each with {self.num_runs} runs - this might take a while..."
                )
        except MemoryError:
            warnings.warn(
                    f"So many configs are being created that it caused a MemoryError. Did you configure this experiment correctly?"
                )
            total_configs = None
        return total_configs, key_paths, param_lists

    def set_global_rng(self, num_seed):
        # Create the RandomState instance
        self.global_rng = np.random.RandomState(num_seed)
        # Give access to it from the classes
        Cluster.global_rng = self.global_rng
        Dataset.global_rng = self.global_rng
        Genotype.global_rng = self.global_rng        

    def run(self):
        raise NotImplementedError

    def get_best_dataset(self, return_config=False):
        raise NotImplementedError
    
    @staticmethod
    def _compare_individuals(indiv1, indiv2):
        raise NotImplementedError

    def plot_best_indiv(self, cmap="rainbow", fig_format="png"):
        if self.best_indiv is None:
            raise ValueError(f"No best individual is stored - have you run the generator?")
        # Loop over the best indiv(s) (will just be 1 in single config case)
        for config_id, best_indiv in enumerate(self.best_indiv):
            fname = self.base_folder / "plots" / f"{config_id}_best_plot"
            # Plot the individual
            plotting.plot_indiv(
                best_indiv,
                save=self.save_plot,
                fname=fname,
                cmap=cmap,
                fig_format=fig_format,
                global_seed=self.seed_num
            )


class SingleObjective(BaseGenerator):
    def __init__(self, params, any_saving, multi_config):
        super().__init__(params, any_saving, multi_config)

    def run(self):
        # Setup a dataframe for storing stats
        self.stats = pd.DataFrame()
        # Get the number of configs for tqdm (and specifics for multi-config)
        if self.multi_config:
            # Initialize the container for the configs
            self.config_list = []
            # Count the number of configs to be, and get the changing params
            total_configs, key_paths, param_lists = self._count_multiconfigs()
        else:
            total_configs, key_paths, param_lists = 1, None, None
        # Create a seed if one was not provided
        if self.seed_num is None:
            self.seed_num = datetime.now().microsecond
            self.full_config["hawks"]["seed_num"] = self.seed_num
        # Save the full config if any saving is happening
        if self.any_saving:
            # Save the now completed full config
            self.save_config(self.full_config, folder=self.base_folder)
        # Initialize the config_id
        config_id = 0
        # Loop over each config
        for params, config in tqdm(self._get_configs(key_paths, param_lists), desc="Configs", total=total_configs):
            # If there are multiple configs, store the individual ones in a list
            if self.multi_config:
                # Add the config to the list
                self.config_list.append(config)
            # Local ref to best for each config
            best_indiv = None
            # Setup the containers for storing results
            num_rows = config["ga"]["num_indivs"]
            results_dict = defaultdict(list)
            # Add the config_id, which is also used for the filename when saving the config
            results_dict["config_id"] = [config_id]*(num_rows*config["ga"]["num_gens"]*self.num_runs)
            # Add the specific parameters for this config
            if self.multi_config:
                for key, param in zip(key_paths, params):
                    name = "_".join(key[1:])
                    results_dict[name] += [param]*(num_rows*config["ga"]["num_gens"]*self.num_runs)
            # Loop over each run
            for num_run in tqdm(range(self.num_runs), desc="Runs", leave=False):
                # Super special seed selection
                global_seed = self.seed_num + (num_run * 10)
                # Create the RandomState instance
                self.set_global_rng(global_seed)
                # Create the Dataset instance
                dataset_obj = prepare.setup_dataset(config["dataset"])
                # Setup the GA
                objective_dict, ga_params, toolbox, pop = prepare.setup_ga(
                    config["ga"],
                    config["constraints"],
                    config["objectives"],
                    dataset_obj
                )
                # Store results from the initial population
                results_dict = self._store_results(
                    results_dict, pop, num_run, 0, num_rows, objective_dict
                )
                # Go through each generation
                for gen in tqdm(
                        range(1, ga_params["num_gens"]),
                        desc="Generations", leave=False
                    ):
                    pop = ga.generation(
                        pop,
                        toolbox,
                        config["constraints"]
                    )
                    # Store results from each generation
                    results_dict = self._store_results(
                        results_dict, pop, num_run, gen, num_rows, objective_dict
                    )
                # Get the best indiv for this particular run
                # Use max with the weighted values to generalize to minimization and maximation (in the single-objective case)
                best_indiv_run = max(pop, key=lambda x: x.fitness.wvalues)
                # Compare to the current best if there is one
                if best_indiv is None:
                    best_indiv = best_indiv_run
                else:
                    best_indiv = self._compare_individuals(
                        best_indiv, best_indiv_run
                    )
            self.best_indiv.append(best_indiv)
            # Iterate the config_id
            config_id += 1
            # Append the results of this config to the overall results
            self.stats = self.stats.append(
                pd.DataFrame.from_dict(results_dict), ignore_index=True
            )
            # Save after each config just in case
            if self.save_stats:
                # Save to CSV
                self.save_stats_csv(
                    self.base_folder / "hawks_stats.csv"
                )
            # Save the best individual(s) and their associated config(s)
            if self.save_best_data:
                # Make a folder for the datasets
                dataset_folder = self.base_folder / "datasets"
                # Save the best data
                best_indiv.save_clusters(dataset_folder, f"{config_id}_best_data")
                # If there are multiple configs then save the specific one for each of the best datasets
                if self.multi_config:
                    # Save the associated config
                    self.save_config(config, folder=dataset_folder, filename=f"{config_id}_config")
        # Plot the best for each config
        if self.plot_best:
            self.plot_best_indiv()
        print("\nSuccess!")
    
    def get_best_dataset(self, return_config=False):
        """
        Function for getting the data and labels of the best dataset for everyone run per config.

        In the single config case, the data and labels of the best individual are returned, and the config if specified.

        In the multi-config case, a list of these things are returned instead, for each of the configs specified. Returning the config is advised here so it is clear what parameters are associated with the returned data and labels of the same index.
        """
        # Check if multi_config or not
        if self.multi_config:
            # Get the dataset for the best_indiv for each config
            best_datasets = [indiv.all_values for indiv in self.best_indiv]
            # Same for labels
            best_labels = [indiv.labels for indiv in self.best_indiv]
            # Return the data, labels and config if specified
            if return_config:
                return best_datasets, best_labels, self.config_list
            else:
                return best_datasets, best_labels
        else:
            if return_config:
                return self.best_indiv[0].all_values, self.best_indiv[0].labels, self.full_config
            else:
                return self.best_indiv[0].all_values, self.best_indiv[0].labels

    def _store_results(self, results_dict, pop, num_run, gen, num_rows, objective_dict):
        # Add some constants for this run
        results_dict["run"] += [num_run]*num_rows
        results_dict["indiv"] += list(range(num_rows))
        results_dict["gen"] += [gen]*num_rows
        # Store the results for each individual in the pop
        for indiv in pop:
            for i, obj_name in enumerate(objective_dict):
                results_dict[obj_name] += [getattr(indiv, obj_name)] 
                results_dict[f"fitness_{obj_name}"] += [indiv.fitness.values[i]]
            for constraint, value in indiv.constraints.items():
                results_dict[constraint] += [value]
        return results_dict

    def _compare_individuals(self, indiv1, indiv2):
        # Go through each tiebreaker for indiv1
        indiv1_results = (
            indiv1.fitness.dominates(indiv2.fitness),
            indiv1.penalty < indiv2.penalty,
            self.global_rng.rand()
        )
        # Go through each tiebreaker for indiv2
        indiv2_results = (
            indiv2.fitness.dominates(indiv1.fitness),
            indiv2.penalty < indiv1.penalty,
            self.global_rng.rand()
        )
        # Return the best individual
        if indiv1_results > indiv2_results:
            return indiv1
        else:
            return indiv2