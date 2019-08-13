"""
Functions to help with error handling and generally support everything else. Could be integrated into the Generator, but sometimes can be used outside that.
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

def get_key_paths(d, key_paths=[], param_lists=[], acc=[]):
    """Used to traverse a config and identify where multiple parameters are given
    """
    for k, v in d.items():
        if isinstance(v, dict):
            get_key_paths(v, key_paths, param_lists, acc=acc + [k])
        elif isinstance(v, list):
            key_paths.append(acc + [k])
            param_lists.append(v)
    return key_paths, param_lists

def set_key_path(d, key_path, v):
    """Used to set the parameter of a multiconfig to a single, given value
    """
    d1 = reduce(dict.get, key_path[:-1], d)
    d1[key_path[-1]] = v

def df_to_csv(df, path, filename):
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
    """Used to get get and format current date, to name folders when no name is given
    """
    return datetime.today().strftime('%Y_%m_%d-%H%M%S')
