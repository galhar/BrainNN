# Writer: Gal Harari
# Date: 30/11/2020
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.Identity_task.main_identity import identity_evaluation
from src.Fonts_task.main_font import fonts_trainer_evaluation, mnist_output_dist, \
    mnist_train_evaluate
from src.binary_encoding_task.main_binary import output_distribution_query, \
    one_one_evaluation
from src.utils.general_utils import save_json
import argparse

SAVE_PATH = "tmp/"
SAVE_NAME = "single_run_data"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("save_idx", help="saves data with the given idx in the name")
    args = parser.parse_args()
    single_run_data = fonts_trainer_evaluation()
    save_name =  SAVE_PATH + SAVE_NAME + args.save_idx
    save_json(single_run_data, save_name)
    print("Saved single run data as: %s" % save_name)
