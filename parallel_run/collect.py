# Writer: Gal Harari
# Date: 30/11/2020
import os

from src.assesing import average_over_nets
from src.utils.general_utils import load_json
from single_run import SAVE_PATH

if __name__ == '__main__':
    # iterate over all the saved data and combine it
    combined_data = []
    print("Collecting...")
    for filename in os.listdir(SAVE_PATH):
        print("Collect data %s" % filename)
        combined_data.append(load_json(filename))
        os.remove(filename)

    print("Processing merged data...")
    average_over_nets(None, load=combined_data)
