# Writer: Gal Harari
# Date: 30/11/2020
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.main_binary import trainer_evaluation
from src.utils.general_utils import save_json
import argparse

SAVE_PATH = "tmp/"
SAVE_NAME = "single_run_data"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("save_idx", help="saves data with the given idx in the name")
    args = parser.parse_args()
    single_run_data = trainer_evaluation()
    save_json(single_run_data, SAVE_PATH + SAVE_NAME + args.save_idx)
