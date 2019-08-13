import functools
from datetime import datetime
from pathlib import Path
from itertools import zip_longest
import warnings
import inspect
# from inspect import getfullargspec

import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.mixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import hawks.utils
import hawks.problem_features

warnings.filterwarnings(
    action='ignore', category=FutureWarning, module="sklearn"
)

def define_cluster_algs(seed):
    cluster_algs = {
        "K-Means++": {
            "class": getattr(sklearn.cluster, "KMeans"),
            "kwargs": {
                "n_clusters": None,
                "random_state": seed,
                "n_init": 10
            }
        },
        "Single-Linkage": {
            "class": getattr(sklearn.cluster, "AgglomerativeClustering"),
            "kwargs": {
                "n_clusters": None,
                "linkage": "single"
            }
        },
        "Average-Linkage": {
            "class": getattr(sklearn.cluster, "AgglomerativeClustering"),
            "kwargs": {
                "linkage": "average",
                "n_clusters": None
            }
        },
        "Single-Linkage (Double)": {
            "class": getattr(sklearn.cluster, "AgglomerativeClustering"),
            "kwargs": {
                "linkage": "single",
                "n_clusters": 2.0
            }
        },
        "Average-Linkage (Double)": {
            "class": getattr(sklearn.cluster, "AgglomerativeClustering"),
            "kwargs": {
                "linkage": "average",
                "n_clusters": 2.0
            }
        },
        "GMM": {
            "class": getattr(sklearn.mixture, "GaussianMixture"),
            "kwargs": {
                "n_components": None,
                "random_state": seed,
                "n_init": 5
            }
        }
    }
    return cluster_algs

def run_clustering(generator=None, datasets=None, label_sets=None, subset=None, save=True, seed=None, df=None, source="hawks", problem_features=False):
    # Something needs to be given
    if generator is None and datasets is None:
        raise ValueError(f"No generator or datasets have been given - there's nothing to evaluate!")
    # If save is true but no generator, need to set a save location
    if save is True and (generator is None or not generator.any_saving):
        base_folder = Path.cwd() / "hawks_experiments" / f"clustering_{hawks.utils.get_date()}"
        # No folder, so create one in similar way to animate?
    elif generator is not None and (save is True or generator.any_saving):
        base_folder = generator.base_folder
    # Extract the datasets and labels from the generator
    if generator is not None:
        datasets, label_sets = generator.get_best_dataset()
        # Get a flat list of the config id for each one of the datasets
        config_nums = np.arange(
            len(generator.best_each_run)
        ).repeat(
            generator.full_config["hawks"]["num_runs"]
        ).tolist()
    # Otherwise just set the config number to be None's
    else:
        config_nums = [None]*len(datasets)
    # Get the problem feature functions if needed
    if problem_features:
        feature_funcs = dict(
            inspect.getmembers(hawks.problem_features, inspect.isfunction)
        )
    # Set the seed used for stochastic algs
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
    # Get the defined clustering algs
    cluster_algs = define_cluster_algs(seed)
    # If a subset of algorithms is given, then select only those
    if subset is not None:
        alg_dict = {}
        for alg_name in subset:
            try:
                alg_dict[alg_name] = cluster_algs[alg_name]
            except KeyError as e:
                raise Exception(f"{alg_name} cannot be found, must be in: {cluster_algs.keys()}") from e
    # Otherwise we are running all defined algs
    else:
        alg_dict = cluster_algs
    # If not dataframe is given, create a new one
    if df is None:
        # Initialize the dataframe
        df = pd.DataFrame()
    # Loop over the datasets
    for dataset_num, (data, labels, config_num) in enumerate(zip_longest(datasets, label_sets, config_nums)):
        # Calculate problem_features if need be
        if problem_features:
            problem_feature_vals = calc_problem_features(data, labels, feature_funcs)
        # Loop over the dict of clustering algorithms
        for name, d in alg_dict.items():
            # Add in the number of clusters
            d["kwargs"] = determine_num_clusters(name, d["kwargs"], labels)
            # Pass the kwargs to the relevant algorithm class
            alg = d["class"](**d["kwargs"])
            # Run the algorithm
            alg.fit(data)
            # Predict labels and compare if we have the truth
            if labels is not None:
                # Obtain labels for this algorithm on this dataset
                if hasattr(alg, "labels_"):
                    labels_pred = alg.labels_.astype(np.int)
                else:
                    labels_pred = alg.predict(data)
                ari_score = adjusted_rand_score(labels, labels_pred)
                ami_score = adjusted_mutual_info_score(labels, labels_pred)
            # No labels, so just set scores to NaN
            else:
                ari_score = np.nan
                ami_score = np.nan
            # Store the specific info for this dataset, for this algorithm
            d = {
                "source": source,
                "config_num": config_num,
                "dataset_num": int(dataset_num),
                "cluster_alg": name,
                "ari": ari_score,
                "ami": ami_score
            }
            # Add the problem feature values if needed
            if problem_features:
                d.update(problem_feature_vals)
            # Calculate evaluation metrics and add to df
            df = df.append(
                d,
                ignore_index=True
            )
    # Save the results if specified
    if save:
        hawks.utils.df_to_csv(
            df=df,
            path=base_folder,
            filename="clustering_results"
        )
    return df

def determine_num_clusters(col_name, alg_kwargs, labels):
    # Fix annoying inconsistency with sklearn arg names
    if col_name == "GMM":
        arg = "n_components"
    else:
        arg = "n_clusters"
    # Check that this alg takes arg as input
    if arg in alg_kwargs:
        # Calc the actual number of clusters
        num_clusts = np.unique(labels).shape[0]
        # Set this as the target
        if alg_kwargs[arg] is None:
            alg_kwargs[arg] = int(num_clusts)
        # Use a multiplier if given
        elif isinstance(alg_kwargs[arg], float):
            multiplier = alg_kwargs[arg]
            alg_kwargs[arg] = int(num_clusts * multiplier)
    return alg_kwargs

def calc_problem_features(data, labels, feature_funcs):
    problem_feature_vals = {}
    # Calculate the feature values for this problem/data
    for name, func in feature_funcs:
        problem_feature_vals[f"f_{name}"] = func(data, labels)
    return problem_feature_vals
