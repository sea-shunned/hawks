from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd

import hawks.generator

def load_datasets(folder_path, glob_filter="*.csv", labels_last_column=True, labels_filename=False, custom_func=None, **kwargs):
    """Function to load datasets from an external source. The path to the folder is given, and by default all .csvs are used. The labels for the data can be specified as a separate file, or final column of the data.

    Any extra kwargs are passed to np.loadtxt, which loads the data in.

    Arguments:
        folder_path {str,Path} -- Path to the folder containing the data

    Keyword Arguments:
        glob_filter {str} -- Select the files in the folder using this filter (default: {"*.csv"})
        labels_last_column {bool} -- If the labels are in the last column or not (default: {True})
        labels_filename {bool} -- If the labels are in a separate file (with 'labels' in the filename) (default: {False})
        custom_func {func} -- Function for processing the data directly, useful if it's a special case. Must return filenames, datasets, and corresponding labels.

    Returns:
        filenames {list} -- A list of the filenames for each loaded file
        datasets {list} -- A list of the loaded datsets
        label_sets {list} -- A list of the labels
    """
    # Convert the folder_path to a Path if needed
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    elif isinstance(folder_path, Path):
        pass
    else:
        raise TypeError(f"{type(folder_path)} is not a valid type for the folder_path")
    # Check that the directory exists
    if not folder_path.is_dir():
        raise ValueError(f"{folder_path} is not a directory")
    # If both labels arguments are true, raise error
    if labels_last_column and labels_filename:
        raise ValueError(f"labels_last_column and labels_filename cannot both be True")
    # or if both are False
    elif not (labels_last_column or labels_filename):
        raise ValueError(f"labels_last_column and labels_filename cannot both be False")
    # Get the files according to the filter provided
    files = list(folder_path.glob(glob_filter))
    # Sort them files (avoids needing leading 0s)
    # https://stackoverflow.com/a/36202926/9963224
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', str(var))])
    # If no files are found, raise error
    if not files:
        raise ValueError(f"{folder_path} with {glob_filter} filter had no results")
    # Run the custom function if provided
    if custom_func is not None:
        filenames, datasets, label_sets = custom_func(files)
    else:
        # Initialize containers
        filenames = []
        datasets = []
        label_sets = []
        # Loop through the files
        for file in files:
            filenames.append(file.name)
            # If the labels are in separate files, load them
            if labels_filename:
                if "label" in file:
                    label_sets.append(np.loadtxt(file, **kwargs))
                else:
                    datasets.append(np.loadtxt(file, **kwargs))
            # Otherwise the labels are in the last column
            elif labels_last_column:
                # Load the data
                data = np.loadtxt(file, **kwargs)
                # Add the datasets and labels
                label_sets.append(data[:, -1].astype(int))
                datasets.append(data[:, :-1])
    # Return the filenames, datasets, and labels
    return filenames, datasets, label_sets

def load_folder(folder_path):
    """Creates a generator object from a folder (previously created by the generator).

    Arguments:
        folder_path {str, Path} -- Name or path to a previously generated folder

    Returns:
        BaseGenerator -- A generator of the subclass specified in the config
    """
    # If it's not a Path, make it one
    if not isinstance(folder_path, Path):
        folder_path = Path(folder_path)
    # If it's not a directory, we can't do anything
    if not folder_path.is_dir():
        raise ValueError(f"{folder_path} is not an existing folder")
    # First select the config
    config_path = list(folder_path.glob("*_config.json"))
    if len(config_path) > 1:
        raise ValueError("More than one config found - unsure which is the main one.")
    elif not config_path:
        raise ValueError("No config found - was one not saved?")
    else:
        config_path = config_path[0]
    # Create the generator object
    gen = hawks.generator.create_generator(config_path)
    # Set the base_folder to be the folder we're in
    gen.base_folder = folder_path
    # Setup the generator
    _, key_paths, param_lists = gen._setup()
    # Get all the config(s)
    for _, config in gen._get_configs(key_paths, param_lists):
        gen.config_list.append(config)
    # Then select the stats
    stats_path = list(folder_path.glob(f"hawks_stats.csv"))
    if len(stats_path) > 1:
        raise ValueError("More than one stats csv found, unsure which is the main one.")
    else:
        # Try to load the stats
        try:
            stats_path = stats_path[0]
            # Load the stats CSV
            gen.stats = pd.read_csv(
                stats_path,
                index_col=False
            )
        except IndexError:
            warnings.warn(
                message=f"No stats csv was found",
                category=UserWarning
            )
            # Set as None
            gen.stats = None
    # Load the datasets
    dataset_paths = list(folder_path.glob("datasets/*"))
    # Check if there is actually anything to load
    if dataset_paths:
        # Load the datasets in
        _, gen.datasets, gen.label_sets = load_datasets(folder_path/"datasets", delimiter=",")
    else:
        warnings.warn(
            message=f"No datasets were found to load",
            category=UserWarning
        )
    return gen
