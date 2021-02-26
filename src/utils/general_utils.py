# Writer: Gal Harari
# Date: 23/11/2020
import json, pickle
import os

import numpy as np

JSON_SUFFIX = '.json'


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def save_json(data, name):
    """
    saves a mix of json compatible objects and np arrays into json by converting
    anything to pure python objects
    :param data:
    :param name: name to save as without suffix
    :return:
    """
    save_name = name + ("" if JSON_SUFFIX in name else JSON_SUFFIX)
    with open(save_name, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)


def load_json(name):
    """
    loads json into json compatible objects (ndarrays saved will become lists)
    :param name: name to load from without suffix or with suffix
    :return:
    """
    load_name = name + ("" if JSON_SUFFIX in name else JSON_SUFFIX)
    with open(load_name, "r") as f:
        data = json.load(f)
    return data


def find_save_name(base_name, suffix='.json'):
    if not os.path.isfile(base_name + suffix):
        return base_name
    i = 0
    while os.path.isfile(base_name + '(' + str(i) + ')' + suffix):
        i += 1
    base_name += '(' + str(i) + ')'
    return base_name


def convert_pkl_to_json(name):
    with open(name + ".pkl", "rb") as input_file:
        data = pickle.load(input_file)

    save_json(data, name)


def get_pparent_dir(file_arg):
    currentdir = os.path.dirname(os.path.abspath(file_arg))
    # currentdir = os.path.dirname(os.path.abspath(inspect.getfile(
    # inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    pparentdir = os.path.dirname(parentdir)
    return pparentdir
