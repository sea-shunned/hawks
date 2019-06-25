"""
Functions to help with error handling and generally support everything else. Could be integrated into the Generator, but sometimes can be used outside that.
"""
import json
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
