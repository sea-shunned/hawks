"""Functions to help with error handling and generally support everything else.
"""
import json
from datetime import datetime
from pathlib import Path
from functools import reduce

def load_json(fname, subdict=None):
    try:
        with open(fname) as json_file:
            params = json.load(json_file)
    except ValueError as e:
        raise Exception(f"Unable to load file {fname}") from e
    if subdict is None:
        return params
    else:
        return params[subdict]

def get_key_paths(d, key_paths=None, param_lists=None, acc=None):
    """Used to traverse a config and identify where multiple parameters are given.

    Args:
        d (dict): Config dictionary.
        key_paths (list): The list of keys to the relevant part of the config.
        param_lists (list): The list of multiple parameters specified in the config.
        acc (list): Tracker for the keys.
    """
    # Avoid mutable default argument issue for first call
    if key_paths is None:
        key_paths = []
    if param_lists is None:
        param_lists = []
    if acc is None:
        acc = []
    # Loop over the items
    for k, v in d.items():
        if isinstance(v, dict):
            get_key_paths(v, key_paths, param_lists, acc=acc + [k])
        elif isinstance(v, list):
            key_paths.append(acc + [k])
            param_lists.append(v)
    return key_paths, param_lists

def set_key_path(d, key_path, v):
    """Used to set the parameter of a multi-config to a single, given value.

    Args:
        d (dict): Config dictionary.
        key_path (list): The list of keys to the relevant part of the config.
        v: The value to be inserted into the config. The type depends on the value.
    """
    d1 = reduce(dict.get, key_path[:-1], d)
    d1[key_path[-1]] = v

def df_to_csv(df, path, filename):
    """Save a :class:`pandas.DataFrame` as a CSV file.

    Args:
        df (:class:`pandas.DataFrame`): Dataframe to save.
        path (:obj:`str`, :class:`pathlib.Path`): Path to the folder where to save the CSV.
        filename (str): The name of the file to save.
    """
    # Check that the folder provided is a path
    if isinstance(path, Path):
        # Make the directory if needed
        path.mkdir(parents=True, exist_ok=True)
    else:
        # Make it a path
        path = Path(path)
        # Make it a directory if it's not
        if not path.is_dir():
            path.mkdir(parents=True)
    # Save to csv via pandas
    df.to_csv(
        path / f"{filename}.csv",
        sep=",",
        index=False
    )

def get_date():
    """Used to get get and format current date, to name folders when no name is given.
    """
    return datetime.today().strftime('%Y_%m_%d-%H%M%S')

def translate_method(input_method):
    """Removal of whitespace/miscellaneous characters to smooth out method names.

    Args:
        input_method (str): The name of the method to adjust.

    Returns:
        str: The cleaned method name.
    """
    # Get the lowercase version
    sel_method = input_method.lower()
    # Remove some common characters
    transtable = str.maketrans({
        "-": "",
        "_": "",
        " ": ""
    })
    return sel_method.translate(transtable)
