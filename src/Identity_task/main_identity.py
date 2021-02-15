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


def create_trainer(epochs=17):
    data_loader = IdentityDataLoader()
    inp_len = len(data_loader.samples[0])
    output_shape = len(data_loader.classes_neurons)

    fc = [BrainNN.FC]
    nodes_details = [inp_len, output_shape * 2, output_shape]
    IINs_details = [(4, ), (4, ), (4, )]
    conn_mat = [[fc, fc, None],
                [None, fc, fc],
                [None, fc, fc]]
    # nodes_details = [inp_len, output_shape]
    # IINs_details = [(4, ),  (4, )]
    # conn_mat = [[fc, fc],
    #             [None, fc]]

    vis_str = 'None'
    iins_factor = 20
    into_iins_factor = 20
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.CONNECTIONS_MAT: conn_mat,
                          BrainNN.IINS_STRENGTH_FACTOR: iins_factor,
                          BrainNN.INTO_IINS_STRENGTH_FACTOR: into_iins_factor,
                          BrainNN.VISUALIZATION_FUNC_STR: vis_str}

    net = BrainNN(configuration_args)
    optimizer = DefaultOptimizer(net=net, epochs=epochs, sample_reps=14, sharp=True,
                                 dec_prob=1, inc_prob=1)
    trainer = Trainer(net, data_loader, optimizer, verbose=True)
    return net, trainer


def identity_evaluation(epochs=6):
    net, trainer = create_trainer(epochs)
    trainer.register_hook(
        lambda trainer: ClassesEvalHook(trainer, IdentityDataLoader(batched=True),
                                        vis_last_ep=True))
    trainer.train()
    net.plot_history()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


if __name__ == '__main__':
    print(identity_evaluation(epochs=4))
