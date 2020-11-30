# Writer: Gal Harari
# Date: 23/11/2020
import json, pickle
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
    with open(name + JSON_SUFFIX, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)


def load_json(name):
    """
    loads json into json compatible objects (ndarrays saved will become lists)
    :param name: name to load from without suffix
    :return:
    """
    with open(name + JSON_SUFFIX, "r") as f:
        data = json.load(f)
    return data


def convert_pkl_to_json(name):
    with open(name + ".pkl", "rb") as input_file:
        data = pickle.load(input_file)

    save_json(data, name)
