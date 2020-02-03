"""Defines the clustering algorithms and handles running them. Primarily used for analysis and instance space generation.
"""
from collections import defaultdict
from pathlib import Path
from itertools import zip_longest
import warnings
import inspect

import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.mixture
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform

import hawks.utils
import hawks.problem_features

warnings.filterwarnings(
    action='ignore', category=FutureWarning, module="sklearn"
)

def define_cluster_algs(seed):
    """Defines some default clustering algorithms. Currently uses four simple algorithms: average-linkage, GMM, K-Means++, and single-linkage.

    Args:
        seed (int): Random seed given to the algorithms. ``int`` is generally fine, but depends on the algorithm implementation.

    Returns:
        dict: A dict where each key is the name of the algorithm, with ``"class"`` as a callable to create (and fit) the model, any ``"kwargs"`` it needs, and ``"k_multiplier"`` if anything other than the true number of clusters is desired.

    .. todo::
        Extend functionality for arbitrary clustering algorithms
    """
    cluster_algs = {
        "Average-Linkage": {
            "class": getattr(sklearn.cluster, "AgglomerativeClustering"),
            "kwargs": {
                "linkage": "average",
                "n_clusters": None
            },
            "k_multiplier": None
        },
        "Average-Linkage (2K)": {
            "class": getattr(sklearn.cluster, "AgglomerativeClustering"),
            "kwargs": {
                "linkage": "average",
                "n_clusters": None
            },
            "k_multiplier": 2.0
        },
        "GMM": {
            "class": getattr(sklearn.mixture, "GaussianMixture"),
            "kwargs": {
                "n_components": None,
                "random_state": seed,
                "n_init": 3
            },
            "k_multiplier": None
        },
        "K-Means++": {
            "class": getattr(sklearn.cluster, "KMeans"),
            "kwargs": {
                "n_clusters": None,
                "random_state": seed,
                "n_init": 10
            },
            "k_multiplier": None
        },
        "Single-Linkage": {
            "class": getattr(sklearn.cluster, "AgglomerativeClustering"),
            "kwargs": {
                "linkage": "single",
                "n_clusters": None
            },
            "k_multiplier": None
        },
        "Single-Linkage (2K)": {
            "class": getattr(sklearn.cluster, "AgglomerativeClustering"),
            "kwargs": {
                "linkage": "single",
                "n_clusters": None
            },
            "k_multiplier": 2.0
        }
    }
    return cluster_algs

def extract_datasets(generator=None, datasets=None, label_sets=None):
    # Something needs to be given
    if generator is None and datasets is None:
        raise ValueError(f"No generator or datasets have been given - there's nothing to evaluate!")
    # Extract the datasets and labels from the generator
    if generator is not None:
        # Create local references to datasets and label_sets
        datasets, label_sets, configs = generator.get_best_dataset(return_config=True)
        # Get a flat list of the config id for each one of the datasets
        config_nums = np.arange(
            len(configs)
        ).repeat(
            generator.full_config["hawks"]["num_runs"]
        ).tolist()
    # Otherwise just set the config number to be None's
    else:
        config_nums = [None]*len(datasets)
    # Test for unequal number of datasets and label sets
    if len(datasets) != len(label_sets):
        raise ValueError("The number of datasets is not equal to the number of labels")
    return datasets, label_sets, config_nums

def setup_folder(save_folder, generator):
    # Prioritize a given save folder
    if save_folder is not None:
        base_folder = Path(save_folder)
    # Or use the generator's folder
    elif generator is not None and generator.base_folder is not None:
        base_folder = generator.base_folder
    # Use current date in cwd as last resort
    else:
        base_folder = Path.cwd() / f"clustering_{hawks.utils.get_date()}"
    return base_folder

def analyse_datasets(generator=None, datasets=None, label_sets=None, cluster_subset=None, feature_subset=None, seed=None, source="HAWKS", prev_df=None, clustering=True, feature_space=True, save=True, save_folder=None, filename="dataset_analysis"):
    """Function to analyze the datasets, either by their :py:mod:`~hawks.problem_features`, clustering algorithm performance, or both.

    Args:
        generator (:class:`~hawks.generator.BaseGenerator`, optional): HAWKS generator instance (that contains datasets). Defaults to None.
        datasets (list, optional): A list of the datasets to be examined. Defaults to None.
        label_sets (list, optional): A list of labels that match the list of datasets. Defaults to None.
        cluster_subset (list, optional): A list of clustering algorithms to use. Defaults to None, where all default clustering algorithms (specified in `:func:~hawks.analysis.define_cluster_algs`) are used.
        feature_subset (list, optional): A list of problem features to use. Defaults to None, where all problem features (specified in `:mod:~hawks.problem_features`) are used.
        seed (int, optional): Random seed number. Defaults to None, where it is randomly selected.
        source (str, optional): Name of the set of datasets. Useful for organizing/analyzing/plotting results. Defaults to "HAWKS".
        prev_df (:py:class:`~pandas.DataFrame`, optional): Pass in a previous DataFrame, with which the results are added to. Defaults to None, creating a blank DataFrame.
        clustering (bool, optional): Whether to run clustering algorithms on the datasets or not. Defaults to True.
        feature_space (bool, optional): Whether to run the problem features on the datasets or not. Defaults to True.
        save (bool, optional): Whether to save the results or not. Defaults to True.
        save_folder ([type], optional): Where to save the results. Defaults to None, where the location of the :class:`~hawks.generator.BaseGenerator` is used. If no :class:`~hawks.generator.BaseGenerator` instance was given, create a folder in the working directory. 
        filename (str, optional): Name of the CSV file to be saved. Defaults to "dataset_analysis".

    Returns:
        (tuple): 2-element tuple containing:

            :py:class:`~pandas.DataFrame`: DataFrame with results for each dataset.

            :py:class:`pathlib.Path`: The path to the folder where the results are saved.
    """
    if clustering is False and feature_space is False:
        raise ValueError("At least one of `clustering` or `feature_space` must be selected, otherwise there is nothing to do")
    # Extract the datasets
    datasets, label_sets, config_nums = extract_datasets(
        generator=generator,
        datasets=datasets,
        label_sets=label_sets
    )
    # Setup the save folder
    if save or save_folder is not None:
        base_folder = setup_folder(save_folder, generator)
        # If a path is given for the save folder, assume saving is wanted
        save = True
    else:
        base_folder = None
    # Initialize the dataframe
    df = pd.DataFrame()
    # Provided seed has priority, then seed from generator
    if seed is None and generator is not None:
        seed = generator.seed_num
    # Otherwise random seed, but raise warning due to unreliable reproducibility
    elif seed is None and generator is None:
        seed = np.random.randint(100)
        warnings.warn(
            message=f"No seed was provided, using {seed} instead",
            category=UserWarning
        )
    # Setup and run feature space functions
    if feature_space:
        # Get the functions from problem_features.py (not imported)
        feature_funcs = dict(
            [func_tup for func_tup in inspect.getmembers(hawks.problem_features, inspect.isfunction) if func_tup[1].__module__ == "hawks.problem_features"]
        )
        # If a feature subset has been given, remove those functions
        if feature_subset is not None:
            feature_dict = {}
            for feature_name in feature_subset:
                try:
                    feature_dict[feature_name] = feature_funcs[feature_name]
                except KeyError as e:
                    raise Exception(f"{feature_name} cannot be found, must be in: {feature_funcs.keys()}") from e
        else:
            feature_dict = feature_funcs
        feature_df = run_feature_space(datasets, label_sets, config_nums, feature_dict, df, source)
    # Setup and run clustering algorithms
    if clustering:
        # Get the defined clustering algs
        cluster_algs = define_cluster_algs(seed)
        # If a subset of algorithms is given, then select only those
        if cluster_subset is not None:
            alg_dict = {}
            for alg_name in cluster_subset:
                try:
                    alg_dict[alg_name] = cluster_algs[alg_name]
                except KeyError as e:
                    raise Exception(f"{alg_name} cannot be found, must be in: {cluster_algs.keys()}") from e
        else:
            alg_dict = cluster_algs
        # Run the clustering algorithms
        cluster_df = run_clustering(datasets, label_sets, config_nums, alg_dict, df, source)
    # Join the dataframes if need be
    if feature_space and clustering:
        # Need to merge on source and dataset number
        # Use concat to handle when config_num may be undefined (rather than pd.merge)
        final_df = pd.concat([cluster_df, feature_df], axis=1)
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    elif feature_space:
        final_df = feature_df
    elif clustering:
        final_df = cluster_df
    if prev_df is not None:
        final_df = prev_df.append(
            final_df,
            ignore_index=True,
            sort=False
        )
    # Save the full dataframe
    if save:
        base_folder.mkdir(parents=True, exist_ok=True)
        hawks.utils.df_to_csv(
            df=final_df,
            path=base_folder,
            filename=filename
        )
    return final_df, base_folder

def run_clustering(datasets, label_sets, config_nums, alg_dict, df, source):
    """Function to actually run the clustering algorithms and add results to the DataFrame.

    Args:
        datasets (list, optional): A list of the datasets to be examined. Defaults to None.
        label_sets (list, optional): A list of labels that match the list of datasets. Defaults to None.
        config_nums (list): A list of the config numbers (only relevant for HAWKS, not external datasets). Allows linking of datasets to parameter configuration.
        alg_dict (dict): Dictionary of the clustering algorithms. Defined in `:func:~hawks.analysis.define_cluster_algs`.
        df (:py:class:`~pandas.DataFrame`): DataFrame to add the results to.
        source (str): Name of the set of datasets.
    
    Returns:
        :py:class:`~pandas.DataFrame`: DataFrame with the clustering results.
    """
    # Loop over the datasets
    for dataset_num, (data, labels, config_num) in enumerate(zip_longest(datasets, label_sets, config_nums)):
        # Create the defaultdict
        res_dict = defaultdict(list)
        # Add the constants to the dict
        res_dict["source"].append(source)
        res_dict["config_num"].append(config_num)
        res_dict["dataset_num"].append(dataset_num)
        # Add some extra general info about the dataset here
        res_dict["num_examples"].append(int(data.shape[0]))
        res_dict["num_clusters"].append(int(np.unique(labels).shape[0]))
        # Loop over the dict of clustering algorithms
        for name, d in alg_dict.items():
            # Add in the number of clusters
            d["kwargs"] = determine_num_clusters(name, d["kwargs"], d["k_multiplier"], labels)
            # Increment the seed to avoid pattern in datasets
            if "random_state" in d["kwargs"]:
                d["kwargs"]["random_state"] += 1
            # Pass the kwargs to the relevant algorithm class
            alg = d["class"](**d["kwargs"])
            # Run the algorithm
            alg.fit(data)
            # Predict labels and compare if we have the truth
            if labels is not None:
                # import pdb; pdb.set_trace()
                # Obtain labels for this algorithm on this dataset
                if hasattr(alg, "labels_"):
                    labels_pred = alg.labels_.astype(np.int)
                else:
                    labels_pred = alg.predict(data)
                ari_score = adjusted_rand_score(labels, labels_pred)
            # No labels, so just set scores to NaN
            else:
                ari_score = np.nan
            # Add the cluster name and scores
            res_dict[f"c_{name}"].append(ari_score)
        # Calculate evaluation metrics and add to df
        # Not particularly efficient
        df = df.append(
            pd.DataFrame.from_dict(res_dict),
            ignore_index=True,
            sort=False
        )
    return df

def run_feature_space(datasets, label_sets, config_nums, feature_dict, df, source):
    """Function to actually run the problem features on the datasets and add results to the DataFrame.

    Args:
        datasets (list, optional): A list of the datasets to be examined. Defaults to None.
        label_sets (list, optional): A list of labels that match the list of datasets. Defaults to None.
        config_nums (list): A list of the config numbers (only relevant for HAWKS, not external datasets). Allows linking of datasets to parameter configuration.
        feature_dict (dict): Dictionary of the problem features to be used.
        df (:py:class:`~pandas.DataFrame`): DataFrame to add the results to.
        source (str): Name of the set of datasets.

    Returns:
        :py:class:`~pandas.DataFrame`: DataFrame with the clustering results.
    """
    # Loop over the datasets
    for dataset_num, (data, labels, config_num) in enumerate(zip_longest(datasets, label_sets, config_nums)):
        # Create the defaultdict
        res_dict = defaultdict(list)
        # Add the constants to the dict
        res_dict["source"].append(source)
        res_dict["config_num"].append(config_num)
        res_dict["dataset_num"].append(dataset_num)
        # Add some extra general info about the dataset here
        res_dict["num_examples"].append(int(data.shape[0]))
        res_dict["num_clusters"].append(int(np.unique(labels).shape[0]))
        # Precomputation for problem features (assumes we always use all)
        precomp_dict = {
            "dists_sqeuclidean": squareform(pdist(data, metric="sqeuclidean"))
        }
        # precomp_dict["dists_euclidean"] = np.sqrt(precomp_dict["dists_sqeuclidean"])
        # Calculate the feature values for this problem/data
        for name, func in feature_dict.items():
            res_dict[f"f_{name}"].append(func(data, labels, precomp_dict))
        # Add to dataframe
        # Not particularly efficient
        df = df.append(
            pd.DataFrame.from_dict(res_dict),
            ignore_index=True,
            sort=False
        )
    return df

def determine_num_clusters(col_name, alg_kwargs, multiplier, labels):
    """Function to extract the number of clusters for the dataset (requires labels, this isn't an estimation process).

    Args:
        col_name (str): Name of the algorithm.
        alg_kwargs (dict): Arguments for the clustering algorithm.
        multiplier (float): Multiplier for the number of clusters.
        labels (list): The labels for this dataset. Can be a list or :py:class:`numpy.ndarray`.

    Raises:
        KeyError: Incorrect algorithm name given.

    Returns:
        dict: The algorithm's arguments with the cluster number added.
    """
    # Fix annoying inconsistency with sklearn arg names
    if col_name == "GMM":
        arg = "n_components"
    else:
        arg = "n_clusters"
    # Check that this alg takes arg as input
    if arg in alg_kwargs:
        # Calc the actual number of clusters
        num_clusts = np.unique(labels).shape[0]
        # Set multiplier to 1 if there isn't one
        if multiplier is None:
            multiplier = 1
        # Calculate what will be given to the algorithm
        given_clusts = int(num_clusts * multiplier)
        # Insert the argument
        alg_kwargs[arg] = given_clusts
        # Ensure that the correct number is being inserted
        assert alg_kwargs[arg] == given_clusts
    else:
        raise KeyError(f"{arg} was not found in {col_name}'s kwargs: {alg_kwargs}")
    return alg_kwargs
