# Writer: Gal Harari
# Date: 14/12/2020
from font_prediction import FontDataLoader
from brainNN import BrainNN
from hooks import ClassesEvalHook, SaveByEvalHook
from utils.train_utils import DefaultOptimizer, Trainer

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

TRAIN_DIR = os.path.join(parentdir, 'data/Font images/Calibri Font images/')
TEST_DIR = os.path.join(parentdir, 'data/Font images/Calibri Font images/')


def create_trainer(epoches=17):
    data_loader = FontDataLoader(TRAIN_DIR)
    img_len = len(data_loader.samples[0])
    output_shape = data_loader.classes_neurons[-1]

    nodes_details = [img_len, 10, 10, output_shape]
    IINs_details = [(4, 4), (3, 3), (3, 3), (1, 1)]
    inter_connections = [(False, True), (True, True), (True, True), (True, True)]
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.INTER_CONNECTIONS_PER_LAYER: inter_connections}

    net = BrainNN(configuration_args)
    optimizer = DefaultOptimizer(net=net, epoches=epoches, sample_reps=6)
    trainer = Trainer(net, data_loader, optimizer, verbose=False)
    return net, trainer


def trainer_evaluation(epoches=20):
    net, trainer = create_trainer(epoches)
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, FontDataLoader(
        TEST_DIR, batched=True)))
    trainer.register_hook(lambda trainer: SaveByEvalHook(trainer, req_acc=70))
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[tot_acc_str]]


if __name__ == '__main__':
    print(trainer_evaluation(epoches=2))
