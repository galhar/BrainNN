# Writer: Gal Harari
# Date: 14/12/2020
from src.Fonts_task.font_prediction import FontDataLoader, IMG_SIZE
from src.brainNN import BrainNN
from src.hooks import ClassesEvalHook, SaveByEvalHook
from src.utils.train_utils import DefaultOptimizer, Trainer

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)

TRAIN_DIR = os.path.join(pparentdir, 'data/Font images/Calibri Font images/')
TEST_DIR = os.path.join(pparentdir, 'data/Font images/Calibri Font images/')


def create_trainer(epoches=17):
    data_loader = FontDataLoader(TRAIN_DIR, shuffle=True)
    img_len = len(data_loader.samples[0])
    output_shape = len(data_loader.classes_neurons)

    nodes_details = [img_len, 140, 100, output_shape]
    IINs_details = [(4, 4), (3, 3), (3, 3), (1, 1)]
    inter_connections = [(True, True), (True, True), (True, True), (True, True)]
    img_dim = (IMG_SIZE, IMG_SIZE)
    feedback = False
    iin_factor = 10
    vis_str = 'None'
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.INTER_CONNECTIONS_PER_LAYER: inter_connections,
                          BrainNN.SPACIAL_ARGS: img_dim,
                          BrainNN.FEEDBACK: feedback,
                          BrainNN.IINS_STRENGTH_FACTOR: iin_factor,
                          BrainNN.VISUALIZATION_FUNC_STR: vis_str}

    net = BrainNN(configuration_args)
    optimizer = DefaultOptimizer(net=net, epoches=epoches, sample_reps=6)
    trainer = Trainer(net, data_loader, optimizer, verbose=True)
    return net, trainer


def fonts_trainer_evaluation(epoches=10):
    net, trainer = create_trainer(epoches)
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, FontDataLoader(
        TEST_DIR, batched=True)))
    trainer.register_hook(lambda trainer: SaveByEvalHook(trainer, req_acc=70))
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


if __name__ == '__main__':
    print(fonts_trainer_evaluation(epoches=1))
