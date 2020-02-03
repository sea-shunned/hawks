"""Defines the overarching Generator class for HAWKS. Intended to be the outward-facing class that users interact with, pulling everything else together.
"""
from itertools import product
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from copy import deepcopy
import shutil
import subprocess
import json
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd

import hawks.ga as ga
import hawks.plotting as plotting
import hawks.objectives as objectives
import hawks.utils as utils
from hawks.cluster import Cluster
from hawks.dataset import Dataset
from hawks.genotype import Genotype

warnings.filterwarnings("ignore", module="deap", category=RuntimeWarning)


def create_generator(config=None):
    """Function to create a generator (of the relevant sub-class of :class:`~hawks.generator.BaseGenerator`) to be used to generate datasets. This is preferential to calling the class directly, particularly in future versions of HAWKS.
    
    Args:
        config (dict, str, :class:`~pathlib.Path`, optional): A dictionary or path to a JSON file with the parameters for HAWKS. Defaults to None, whereby the defaults are used.
    
    Raises:
        FileNotFoundError: If a path-like object is given, but the file cannot be found.
        TypeError: If an object is given that cannot be interpreted into a config file (i.e. not a ``dict`` or a path-like object).
        ValueError: If an incorrect mode for HAWKS is provided.
    
    Returns:
        :class:`~hawks.generator.BaseGenerator`: The initialized generator instance.
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
    """Base class for the generator, providing a set of common functionality required for all future classes/functionality. Not all attributes are shown here, but those that are not are derived from the config.

    Attributes:
        config (dict): The full config.
        any_saving (bool): Whether any of the saving options have been specified. This is automatically determined.
        multi_config (bool): Whether the config specifies multiple sets of parameters or not. This is automatically determined.
        stats (:py:class:`~pandas.DataFrame`): A DataFrame that records stats during the runs for easy analysis.
        datasets (list): A list of datasets (extracted arrays from the :class:`~hawks.genotype.Genotype` s). Created in :func:`~hawks.generator.BaseGenerator.get_best_dataset`.
        label_sets (list): A list of labels (extracted arrays from the :class:`~hawks.genotype.Genotype` s). Created in :func:`~hawks.generator.BaseGenerator.get_best_dataset`.
        population (list): An easy reference to the most recent population of individuals.
        base_folder (:class:`~pathlib.Path`): The path to the root folder for this run of HAWKS. Constructued using the ``folder_name`` in the config. If one isn't given, uses that datetime.
        config_list (list): A list of the unique configs. Useful for multi_config, where each combination of parameters gets a single config which is stored in this list.
        best_each_run (list): A list, which contains a sub-list for each config. In this sub-list, the best individual from each run is stored. For single-config runs, flattening may be needed.
    """
    def __init__(self, config, any_saving, multi_config):
        #: Keep the full config
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
        self.comparison = config["hawks"]["comparison"]
        self.save_best_data = config["hawks"]["save_best_data"]
        self.save_stats = config["hawks"]["save_stats"]
        self.save_config_flag = config["hawks"]["save_config"]
        # Initialize attributes
        self.population = None # Reference to the final pop
        # self.best_dataset = None # Array of the best dataset
        # self.best_indiv = None # Best indiv(s), Genotype object
        self.best_each_run = [] # Best indiv from each run
        self.datasets = None # Easy reference to datasets in .best_each_run
        self.label_sets = None # Easy reference to labels in .best_each_run
        # self.best_config = None # Config for the best indiv
        self.base_folder = None # Folder where everything is saved
        self.global_rng = None # RandomState instance used throughout
        self.config_list = [] # Container for storing each of the configs
        self.stats = None # DataFrame of stats stored throughout the run
        self.deap_toolbox = None # Reference to DEAP's toolbox for working with single indivs

    @staticmethod
    def process_config(config):
        # Load the default config
        base_config = BaseGenerator.load_default_config()
        # Check that the config has valid parameter names
        BaseGenerator._check_config(config, base_config)
        # Merge the default values with the supplied config
        full_config = BaseGenerator._merge_default_config(config, base_config)
        # Check if there is a True in any of the saving parameters
        any_saving = any([v for k, v in full_config["hawks"].items() if "save" in k])
        # Check if a folder name is given, and if any saving is occurring
        if full_config["hawks"]["folder_name"] is None:
            # If there is saving, create a name from the time
            if any_saving:
                full_config["hawks"]["folder_name"] = utils.get_date()
        return full_config, any_saving

    @staticmethod
    def _merge_default_config(config, defaults):
        # Loop through defaults
        for key in defaults:
            # Check if the config specifies the key
            if key in config:
                if config[key] is None:
                    continue
                # Recurse if needed
                elif isinstance(defaults[key], dict):
                    BaseGenerator._merge_default_config(
                        config[key], defaults[key]
                    )
            # Add from defaults if not defined
            else:
                config[key] = defaults[key]
        return config

    @staticmethod
    def _check_config(config, defaults, path=None):
        if path is None:
            path = []
        # Loop through given config
        for key in config:
            if isinstance(config, list):
                continue
            # Check that each key is in the defaults
            if key in defaults:
                if config[key] is None:
                    continue
                # Recurse if needed
                elif isinstance(defaults[key], dict):
                    BaseGenerator._check_config(
                        config[key], defaults[key], path + [str(key)]
                    )
            # Raise error if it can't be found
            else:
                raise ValueError(f"{path + [str(key)]} is not a valid argument for the config")

    def create_folders(self):
        """Create the folder(s) necessary for saving. This function is only called if any saving option is switched on.
        """
        # Create the base folder
        self._create_base_folder()
        # Make a datasets folder if needed
        if self.save_best_data:
            Path(self.base_folder / "datasets").mkdir(exist_ok=True)

    def _create_base_folder(self):
        # Create the base folder
        if self.base_folder is None:
            # Create a folder name if there isn't one
            if self.folder_name is None:
                self.folder_name = utils.get_date()
            # Set the path for the base folder
            self.base_folder = Path.cwd() / self.folder_name
        # Print a warning if the folder exists
        if self.base_folder.is_dir():
            warnings.warn(
                message=f"{self.base_folder} already exists - previous results may be overwritten!",
                category=UserWarning
            )
        # Make the folder
        self.base_folder.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def load_default_config():
        """Loads the default config JSON file (which is used to fill in any gaps in the provided config).
        
        Returns:
            dict: The full config.
        """
        default_config = utils.load_json(Path(__file__).parent / "defaults.json")
        return default_config

    def save_config(self, config=None, folder=None, filename=None):
        """Save the config file for future reproduction.

        Args:
            config (dict, optional): HAWKS config. Defaults to None, where the config is taken from the instance.
            folder (str, :class:`pathlib.Path`, optional): The folder to save the config in. Defaults to None, where the directory of the config file/experiment script/working directory is used.
            filename (str, optional): Filename for the config. Defaults to None, where the experiment/folder name from the config is used instead.
        """
        # Determine location of the config
        # If no folder is given, default to the configs folder
        if folder is None:
            if self.base_folder is None:
                folder = Path.cwd()
            else:
                folder = self.base_folder
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
        # If not config is given, save the full one
        if config is None:
            config = self.full_config
        # Save the config
        with open(fpath, "w") as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def _check_multiconfig(config):
        """Check if a list exists in the config i.e. it defines a set of parameters. Switches to multi_config mode if so.

        Args:
            config (dict): HAWKS config.

        Returns:
            bool: ``True`` if a set of parameters is found.

        See Also:
            :ref:`example_multiconfig` example for further information of usage.
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
        """Return the stats (:py:class:`~pandas.DataFrame`), useful in an interactive setting
        """
        return self.stats

    def get_config(self):
        """Return the full config, useful in an interactive setting
        """
        return self.full_config

    def _get_configs(self, key_paths, param_lists):
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
                    message=f"{total_configs} configs will be generated, each with {self.num_runs} runs - this might take a while...",
                    category=UserWarning
                )
        except MemoryError:
            # Should probably raise the error here
            warnings.warn(
                message=f"So many configs are being created that it caused a MemoryError. Did you configure this experiment correctly?",
                category=UserWarning
            )
            total_configs = None
        return total_configs, key_paths, param_lists

    def increment_seed(self, num_run):
        # Super special seed selection
        return self.seed_num + (num_run * 10)

    def set_global_rng(self, num_seed):
        """Sets the global :py:class:`~numpy.random.mtrand.RandomState` instance for the run, allowing reproducability.

        Args:
            num_seed (int): The seed used to initialize the RNG.
        """
        # Create the RandomState instance
        self.global_rng = np.random.RandomState(num_seed)
        # Give access to it from the classes
        Cluster.global_rng = self.global_rng
        Dataset.global_rng = self.global_rng
        Genotype.global_rng = self.global_rng

    def _setup(self):
        # Create a seed if one was not provided
        if self.seed_num is None:
            self.seed_num = datetime.now().microsecond
            self.full_config["hawks"]["seed_num"] = self.seed_num
        # Create folders if required and save full config
        if self.any_saving:
            self.create_folders()
            # Save the now completed full config
            if self.save_config_flag:
                self.save_config(self.full_config, folder=self.base_folder)
        # Setup a dataframe for storing stats
        if self.stats is None:
            self.stats = pd.DataFrame()
        # Get the number of configs for tqdm (and specifics for multi-config)
        if self.multi_config:
            # Count the number of configs to be, and get the changing params
            total_configs, key_paths, param_lists = self._count_multiconfigs()
        else:
            total_configs, key_paths, param_lists = 1, None, None
        return total_configs, key_paths, param_lists

    def setup_ga(self, dataset_obj, config):
        # Validate the constraints parameters
        Genotype.validate_constraints(config["constraints"])
        # Setup the objective parameters
        objective_dict = self._setup_objectives(config)
        # Create the DEAP toolbox and generate the initial population
        toolbox, initial_pop = ga.main_setup(
            objective_dict=objective_dict,
            dataset_obj=dataset_obj,
            ga_params=config["ga"],
            constraint_params=config["constraints"]
        )
        # Set the DEAP toolbox as an attr
        self.deap_toolbox = toolbox
        # Return the objective dict and the initial population
        return objective_dict, initial_pop

    def run(self):
        raise NotImplementedError

    def run_step(self):
        raise NotImplementedError

    def create_individual(self):
        """Hacky function to do the bare minimum needed to create some individuals. The initial population is generated and we yield from that. Useful for debugging, to just playing with a single dataset/individual.
        """
        if self.multi_config:
            raise ValueError(f"Not available in multi_config mdoe - Need a single config to generate an individual from")
        # Create the RandomState instance
        self.set_global_rng(self.seed_num)
        # Create the Dataset instance
        dataset_obj = Dataset(**self.full_config["dataset"])
        # Setup some attributes for the Cluster class
        Cluster.setup_variables(dataset_obj, self.full_config["ga"])
        # Setup the GA
        objective_dict, pop = self.setup_ga(dataset_obj, self.full_config)
        # Take from the initial population
        yield from pop

    def get_best_dataset(self, return_config=False, reset=False):
        """
        Function for extracting the data and labels of the best dataset for every run per config.

        A list of the datasets (numpy arrays) and a list of the labels are returned. If specified, a list of the associated configs are also returned.

        Note that these lists are flattened. In the single config case, the list of datasets will be ``num_runs`` long. In the multi_config case, a list of length ``num_runs``*``len(self.config_list)`` will be returned.

        If the datasets or label_sets have not already been extracted then they are extracted. If this needs to be updated, there is a flag to reset this and extract again.

        Args:
            return_config (bool): Whether the config should be returned.
            reset (bool): Whether to re-initialize the attributes (useful for when interacting with a run).
        """
        # Get the dataset for the best_indiv for each config
        if self.datasets is None or reset:
            self.datasets = [indiv.all_values for config_list in self.best_each_run for indiv in config_list]
        # Same for labels
        if self.label_sets is None or reset:
            self.label_sets = [indiv.labels for config_list in self.best_each_run for indiv in config_list]
        # Return the configs if specified
        if return_config:
            return self.datasets, self.label_sets, self.config_list
        else:
            return self.datasets, self.label_sets

    def _compare_individuals(self, indiv1, indiv2):
        raise NotImplementedError

    def _store_results(self, results_dict, pop, num_run, gen, num_rows, objective_dict):
        raise NotImplementedError
    
    def _store_best_indiv(self, results_dict, best_index, num_gens, num_rows):
        raise NotImplementedError

    def _plot_save_setup(self):
        # Check if the base_folder was ever set
        if self.base_folder is None:
            # Create it
            self._create_base_folder()
        # Create the plots folder
        plot_folder = Path(self.base_folder / "plots")
        plot_folder.mkdir(exist_ok=True, parents=True)
        return plot_folder

    def plot_best_indivs(self, cmap="inferno", fig_format="pdf", save=False, show=True, remove_axis=False, fig_title=None, nrows=None, ncols=None):
        """Plot the best individuals from each run, for each config. A separate plot is made for each config, with the best from each run plotted together.

        Does the processing required to pass onto :func:`~hawks.generator.BaseGenerator.plot_datasets`.

        Args:
            cmap (str, optional): The colourmap from matplotlib to use. Defaults to "inferno".
            fig_format (str, optional): The format to save the plot in, usually either "png" or "pdf". Defaults to "pdf".
            save (bool, optional): Save the plot. Defaults to False.
            show (bool, optional): Show the plot. Defaults to True.
            remove_axis (bool, optional): Whether to remove the axis to just show the clusters. Defaults to False.
            fig_title (str, optional): Figure title. Defaults to None.
            nrows (int, optional): Number of rows for plt.subplots, calculated if None. Defaults to None.
            ncols (int, optional): Number of columns for plt.subplots, calculated if None. Defaults to None.

        Raises:
            ValueError: If there is no best dataset found. The generator may not have been run yet.
        """
        # Raise error if run premmaturely
        if self.best_each_run is None:
            raise ValueError(f"No best individuals are stored - have you run the generator?")
        # Loop over the configs
        for config_id, config_set in enumerate(self.best_each_run):
            # Get the path if saving
            if save:
                filename = f"config-{config_id}_best-indivs"
            else:
                filename = None
            # Plot the indivs for this config
            self.plot_datasets(
                config_set,
                nrows=nrows,
                ncols=ncols,
                filename=filename,
                cmap=cmap,
                fig_format=fig_format,
                save=save,
                show=show,
                remove_axis=remove_axis,
                fig_title=fig_title
            )

    def plot_datasets(self, datasets, cmap="inferno", fig_format="png", save=False, show=True, remove_axis=False, filename=None, fig_title=None, nrows=None, ncols=None, folder=None, **kwargs):
        """Plot a set of datasets.

        Args:        
            datasets (list): A list of individuals (:class:`~hawks.genotype.Genotype`) to be plotted.
            cmap (str): The colourmap from matplotlib to use.  Defaults to "inferno".
            fig_format (str): The format to save the plot in, usually either "png" or "pdf". Defaults to None.
            save (bool): Save the plot. Defaults to False.
            show (bool): Show the plot. Defaults to True.
            remove_axis (bool): Whether to remove the axis to just show the clusters. Defaults to False.
            filename (str): Filename (constructed if None). Defaults to None.
            fig_title (str): Figure title. Defaults to None.
            nrows (int): Number of rows for plt.subplots, calculated if None. Defaults to None.
            ncols (int): Number of columns for plt.subplots, calculated if None. Defaults to None.
        """
        # Sort out folders if saving
        if save:
            if folder is None:
                plot_folder = self._plot_save_setup()
            else:
                plot_folder = folder
        # Get the path if saving
        if save or filename is not None:
            # Construct filename if not supplied
            if filename is None:
                filename = f"datasets_plot"
            # Assume saving is desired if filename is given
            if filename is not None:
                save = True
            # Concatenate whole path
            fpath = plot_folder / filename
        else:
            fpath = None
        # Plot the indivs for this config
        plotting.plot_pop(
            datasets,
            nrows=nrows,
            ncols=ncols,
            fpath=fpath,
            cmap=cmap,
            fig_format=fig_format,
            save=save,
            show=show,
            global_seed=self.seed_num,
            remove_axis=remove_axis,
            fig_title=fig_title,
            **kwargs
        )

class SingleObjective(BaseGenerator):
    """Class specific for optimizing to a single objective. This is currently the main mode, and is used to optimize the datasets towards as given silhouette width (according the constraints and other parameters).
    """
    def __init__(self, config, any_saving, multi_config):
        super().__init__(config, any_saving, multi_config)

    def run(self):
        """The main run function for the generator.
        """
        for _ in self.run_step():
            pass

    def run_step(self):
        """Run function that contains the actual code, yielding after each run, if desired.
        
        Yields:
            :class:`~hawks.generator.SingleObjective`: The generator instance at the time, allowing inspection of the process.
        """
        total_configs, key_paths, param_lists = self._setup()
        # Initialize the config_id
        config_id = 0
        # Loop over each config
        for params, config in tqdm(self._get_configs(key_paths, param_lists), desc="Configs", total=total_configs):
            # Add the config to the list
            self.config_list.append(config)
            # Add a list as container for new runs
            self.best_each_run.append([])
            # Local ref to best for each config
            best_indiv_run = None
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
                # Increment the seed for this run
                global_seed = self.increment_seed(num_run)
                # Create the RandomState instance
                self.set_global_rng(global_seed)
                # Create the Dataset instance
                dataset_obj = Dataset(**config["dataset"])
                # Setup some attributes for the Cluster class
                Cluster.setup_variables(dataset_obj, config["ga"])
                # Setup the GA
                objective_dict, pop = self.setup_ga(dataset_obj, config)
                # Store results from the initial population
                results_dict = self._store_results(
                    results_dict, pop, num_run, 0, num_rows, objective_dict
                )
                # Go through each generation
                for gen in tqdm(
                        range(1, config["ga"]["num_gens"]),
                        desc="Generations", leave=False
                    ):
                    pop = ga.generation(
                        pop,
                        self.deap_toolbox,
                        config["constraints"],
                        cxpb=config["ga"]["mate_prob"]
                    )
                    # Store results from each generation
                    results_dict = self._store_results(
                        results_dict, pop, num_run, gen, num_rows, objective_dict
                    )
                best_indiv_run, best_index = self._best_in_pop(pop)
                # Store the best indiv from each run
                self.best_each_run[-1].append(best_indiv_run)
                # Add column to show best dataset from run
                results_dict = self._store_best_indiv(
                    results_dict, best_index, config["ga"]["num_gens"], num_rows
                )
                # Keep a reference to the most recent population
                self.population = pop
                # YIELDIT
                yield self
            # Iterate the config_id
            config_id += 1
            # Append the results of this config to the overall results
            self.stats = self.stats.append(
                pd.DataFrame.from_dict(results_dict), ignore_index=True
            )
        # Save the stats for this run if specified
        if self.save_stats:
            # Save to CSV
            utils.df_to_csv(
                df=self.stats,
                path=self.base_folder,
                filename="hawks_stats"
            )
        # Save the best individual(s) and their associated config(s)
        if self.save_best_data:
            # Loop over each indiv in each config
            for config_num, indiv_list in enumerate(self.best_each_run):
                for run_num, indiv in enumerate(indiv_list):
                    # Save the best data
                    indiv.save_clusters(
                        folder=self.base_folder / "datasets",
                        fname=f"config-{config_num}_run-{run_num}_best_data"
                    )

    def _setup_objectives(self, config):
        # Get the currently available/implemented objectives
        avail_objectives = {
            cls.__name__.lower():{'class':cls} for cls in objectives.ClusterIndex.__subclasses__()
        }
        # Create a dict to hold the objectives we select
        objective_dict = {}
        # Loop through the specified objectives
        for selected_obj in config["objectives"]:
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
            obj_args = config["objectives"][selected_obj]
            # Just for completeness
            objective_dict[selected_obj]["kwargs"] = obj_args
            # Set the kwargs for the class
            objective_dict[selected_obj]['class'].set_kwargs(obj_args)
        return objective_dict

    def _best_in_pop(self, pop):
        if self.comparison == "ranking":
            best_indiv = pop[0]
            best_index = 0
        elif self.comparison == "fitness":
            # Turn the weighted fitnesses into an array
            # As single-objective, we extract the only value from tuple
            fitnesses = np.array([x.fitness.wvalues[0] for x in pop])
            # Get the indices of the best individuals
            best_indices = np.argwhere(fitnesses == np.max(fitnesses)).flatten().tolist()
            # Select first indiv as the best one (if tied)
            best_indiv = pop[best_indices[0]]
            # If there is more than one, compare with the others
            if len(best_indices) > 1:
                # Iteratively compare against others, taking best from each
                for curr_index in best_indices[1:]:
                    best_indiv = self._compare_individuals(
                        pop[curr_index], best_indiv
                    )
                # Get the index of the best individual
                best_index = pop.index(best_indiv)
            else:
                best_index = best_indices[0]
        else:
            raise ValueError(f"Comparison method '{self.comparison}' not recognized")
        return best_indiv, best_index

    def _best_across_runs(self):
        # Create container
        best_indivs = []
        # Loop through the best from each run for each config
        for best_in_config in self.best_each_run:
            # Select the first indiv
            best_indiv = best_in_config[0]
            # Compare against the others
            for indiv in best_in_config[1:]:
                best_indiv = self._compare_individuals(best_indiv, indiv)
            # Get the index
            index = best_in_config.index(best_indiv)
            best_indivs.append((best_indiv, index))
        return best_indivs

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

    def _store_best_indiv(self, results_dict, best_index, num_gens, num_rows):
        # Create a whole column of 0s
        best_indiv_column = [0]*num_rows*num_gens
        # Best indiv is in last generation only
        best_indiv_column[(num_gens-1)*num_rows+best_index] = 1
        # Add the column to the results_dict
        results_dict["best_indiv"] += best_indiv_column
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

    def animate(self, record_stats=False, plot_pop=True, **kwargs):
        """Function to animate a run of HAWKS (showing how the datasets evolve). An example of this can be found in the README. Produces a series of PNGs, and creates a gif using `ImageMagick <https://imagemagick.org/index.php>`_.
        
        Args:
            record_stats (bool, optional): Whether the results of the run should be recorded (and therefore can be saved, depending on the config). Defaults to False.
            plot_pop (bool, optional): Whether to plot the whole population. If False, just plots the best individual. Defaults to True.
        
        Raises:
            ValueError: Animation cannot be run for a multi_config; only a single set of parameters is permitted.
        """
        # Raise error if multi-config specified
        if self.multi_config:
            raise ValueError(f"Animation is not implemented for multi-config")
        # Perform initial setup
        total_configs, key_paths, param_lists = self._setup()
        # Setup the containers for storing results
        if record_stats:
            num_rows = self.full_config["ga"]["num_indivs"]
            results_dict = defaultdict(list)
        # Setup the plot folder
        plot_folder = self._plot_save_setup()
        # Loop over each run
        for num_run in tqdm(range(self.num_runs), desc="Runs", leave=False):
            animate_folder = plot_folder / f"animate_run{num_run}"
            animate_folder.mkdir(exist_ok=True, parents=True)
            # Super special seed selection
            global_seed = self.increment_seed(num_run)
            # Create the RandomState instance
            self.set_global_rng(global_seed)
            # Create the Dataset instance
            dataset_obj = Dataset(**self.full_config["dataset"])
            # Setup some attributes for the Cluster class
            Cluster.setup_variables(dataset_obj, self.full_config["ga"])
            # Setup the GA
            objective_dict, pop = self.setup_ga(dataset_obj, self.full_config)
            # Plot the initial population
            if plot_pop:
                plotting.plot_pop(
                    pop,
                    fpath=animate_folder / "pop_gen-0",
                    fig_format="png",
                    save=True,
                    remove_axis=True,
                    fig_title="Generation 0",
                    show=False,
                    **kwargs
                )
            # Plot the best indiv
            else:
                best_indiv, best_index = self._best_in_pop(pop)
                plotting.plot_indiv(
                    best_indiv,
                    fpath=animate_folder / "indiv_gen-0",
                    remove_axis=True,
                    save=True,
                    show=False,
                    **kwargs
                )
            if record_stats:
                # Store results from the initial population
                results_dict = self._store_results(
                    results_dict, pop, num_run, 0, num_rows, objective_dict
                )
            # Go through each generation
            for gen in tqdm(
                    range(1, self.full_config["ga"]["num_gens"]),
                    desc="Generations", leave=False
                ):
                pop = ga.generation(
                    pop,
                    self.deap_toolbox,
                    self.full_config["constraints"],
                    cxpb=self.full_config["ga"]["mate_prob"]
                )
                # Plot the initial population
                if plot_pop:
                    plotting.plot_pop(
                        pop,
                        fpath=animate_folder / f"gen-{gen}",
                        fig_format="png",
                        save=True,
                        remove_axis=True,
                        fig_title=f"Generation {gen}",
                        show=False,
                        **kwargs
                    )
                # Plot the best indiv
                else:
                    best_indiv, best_index = self._best_in_pop(pop)
                    plotting.plot_indiv(
                        best_indiv,
                        fpath=animate_folder / f"indiv_gen-{gen}",
                        remove_axis=True,
                        save=True,
                        show=False,
                        **kwargs
                    )
                if record_stats:
                    # Store results from each generation
                    results_dict = self._store_results(
                        results_dict, pop, num_run, gen, num_rows, objective_dict
                    )
                # Keep a reference to the most recent population
                self.population = pop
        if record_stats:
            self.stats = self.stats.append(
                pd.DataFrame.from_dict(results_dict), ignore_index=True
            )
            if self.save_stats:
                # Save to CSV
                utils.df_to_csv(
                    df=self.stats,
                    path=self.base_folder,
                    filename="hawks_stats"
                )
        # Create the gif if convert is available
        which_convert = shutil.which("convert")
        if which_convert is not None:
            subprocess.run(
                "convert -resize 50% -delay 30 -loop 0 `ls -v | grep 'gen-'` hawks_animation.gif",
                shell=True,
                check=True,
                cwd=animate_folder
            )
