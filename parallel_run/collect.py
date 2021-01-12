# Writer: Gal Harari
# Date: 30/11/2020
import os
import sys
import inspect
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.assesing import average_over_nets
from src.utils.general_utils import load_json
from single_run import SAVE_PATH, SAVE_NAME

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("del_tmp",nargs='?', help="If true delete tmp folder",
                        type=int, default=0)
    parser.add_argument("start_idx",nargs='?', help="first idx to collect", type=int,
                        default=0)
    parser.add_argument("end_idx",nargs='?', help="last idx to collect", type=int,
                        default=1000)
    args = parser.parse_args()

    # iterate over all the saved data and combine it
    combined_data = []
    print("Collecting...")
    for filename in os.listdir(SAVE_PATH):
        idx = int(filename.split(SAVE_NAME)[1].split('.json')[0])
        if args.start_idx <= idx <= args.end_idx:
            print("Collect data %s" % filename)
            combined_data.append(load_json(SAVE_PATH + filename))

    print("Processing merged data...")
    average_over_nets(None, load=combined_data)
    if args.del_tmp:
        for filename in os.listdir(SAVE_PATH):
            print("Cleaning tmp...")
            os.remove(SAVE_PATH + filename)
