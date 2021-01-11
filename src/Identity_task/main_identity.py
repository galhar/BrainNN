# Writer: Gal Harari
# Date: 14/12/2020
from src.Identity_task.identity_prediction import IdentityDataLoader, N
from src.brainNN import BrainNN
from src.hooks import ClassesEvalHook, SaveByEvalHook
from src.utils.train_utils import DefaultOptimizer, Trainer

import os
import sys
import inspect
import cv2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

TRAIN_DIR = os.path.join(parentdir, 'data/Font images/Calibri Font images/')
TEST_DIR = os.path.join(parentdir, 'data/Font images/Calibri Font images/')


def create_trainer(epoches=17):
    data_loader = IdentityDataLoader()
    inp_len = len(data_loader.samples[0])
    output_shape = len(data_loader.classes_neurons)

    nodes_details = [inp_len, output_shape]
    IINs_details = [(4, 4), (1, 4)]
    inter_connections = [(False, False), (False, False)]
    feedback = False
    vis_str = 'None'
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.INTER_CONNECTIONS_PER_LAYER: inter_connections,
                          BrainNN.FEEDBACK: feedback,
                          BrainNN.VISUALIZATION_FUNC_STR: vis_str}

    net = BrainNN(configuration_args)
    optimizer = DefaultOptimizer(net=net, epoches=epoches, sample_reps=6)
    trainer = Trainer(net, data_loader, optimizer, verbose=False)
    return net, trainer


def identity_evaluation(epoches=6):
    net, trainer = create_trainer(epoches)
    trainer.register_hook(
        lambda trainer: ClassesEvalHook(trainer, IdentityDataLoader(batched=True)))
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


if __name__ == '__main__':
    print(identity_evaluation(epoches=5))
