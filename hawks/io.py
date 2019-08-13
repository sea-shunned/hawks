from pathlib import Path
import re

import numpy as np

def load_datasets(folder_path, glob_filter="*.csv", labels_last_column=False, labels_filename=True, **kwargs):
    """Function to load datasets from an external source. The path to the folder is given, and by default all .csvs are used. The labels for the data can be specified as a separate file, or final column of the data.

    Any extra kwargs are passed to np.loadtxt, which loads the data in.

    Arguments:
        folder_path {str,Path} -- Path to the folder containing the data

    Keyword Arguments:
        glob_filter {str} -- Select the files in the folder using this filter (default: {"*.csv"})
        labels_last_column {bool} -- If the labels are in the last column or not (default: {False})
        labels_filename {bool} -- If the labels are in a separate file (with 'labels' in the filename) (default: {True})

    Returns:
        filenames {list} -- A list ofr the name for each loaded file
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
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', str(var))])
    # If no files are found, raise error
    if not files:
        raise ValueError(f"{folder_path} with {glob_filter} filter had no results")
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
            label_sets.append(data[:, -1])
            datasets.append(data[:, :-1])
    # Return the datasets and associated labels
    return filenames, datasets, label_sets
