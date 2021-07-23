# Writer: Gal Harari
# Date: 22/07/2021
# Writer: Gal Harari
# Date: 14/12/2020
from src.Fonts_task.font_prediction import FontDataLoader, MNISTDataLoader, IMG_SIZE
from src.brainNN import BrainNN, SAVE_NAME, SAVE_SUFFIX
from src.hooks import ClassesEvalHook, OutputDistributionHook, SaveHook, SaveByEvalHook
from src.utils.train_utils import DefaultOptimizer, Trainer
from src.utils.general_utils import get_pparent_dir

import os
import numpy as np

pparentdir = get_pparent_dir(__file__)

dirs = [
    'data/Font images/Calibri Font images/',
    'data/Font images/Bell MT images/',
    'data/Font images/Bodoni MT images/',
    'data/Font images/Book Antiqua/',
    'data/Font images/Calisto MT images/',
    'data/Font images/Times New Roman Font images/'
]
TRAIN_DIRS = [os.path.join(pparentdir, dirs[i]) for i in range(1, 6)]
TEST_DIR = os.path.join(pparentdir, dirs[0])

LOAD = False


def create_trainer(data_loader, epochs=17):
    img_len = len(data_loader.samples[0])
    output_shape = len(data_loader.classes_neurons)

    fc = [BrainNN.FC]
    kernel = 3
    stride = 1
    into_n = 1
    white = True
    rf = [BrainNN.RF, [kernel, stride, into_n, white]]

    nodes_details = [img_len, output_shape]
    IINs_details = [(1,), (1,)]
    winners = [0, 1]
    conn_mat = [[fc, fc],
                [None, fc]]
    img_dim = (IMG_SIZE, IMG_SIZE)
    spacial_dist_fac = 1.01
    vis_str = 'None'
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.CONNECTIONS_MAT: conn_mat,
                          BrainNN.WINNERS_PER_LAYER: winners,
                          BrainNN.SPACIAL_ARGS: img_dim,
                          BrainNN.SYNAPSE_SPACIAL_DISTANCE_FACTOR: spacial_dist_fac,
                          BrainNN.VISUALIZATION_FUNC_STR: vis_str}

    if LOAD:
        net = BrainNN.load_model(configuration_args, "NetSavedByHookEp-14(0).json")
    else:
        net = BrainNN(configuration_args)

    inc_func = lambda weights: np.minimum(np.minimum(weights / (10 * 5),
                                                     np.exp(-weights) / 5),
                                          0.04 / 5)
    optimizer = DefaultOptimizer(net=net, epochs=epochs, sample_reps=5, sharp=True,
                                 inc_prob=1.0, dec_prob=0.0)
    optimizer.inc_func = inc_func
    trainer = Trainer(net, data_loader, optimizer, verbose=True)
    return net, trainer


def multifonts_evaluation(epochs=3):
    print("[*] Creating the trainer")
    train_loader = FontDataLoader(TRAIN_DIRS, shuffle=True)
    test_loader = FontDataLoader(TEST_DIR, shuffle=True)
    net, trainer = create_trainer(train_loader, epochs)
    trainer.register_hook(
        lambda trainer: SaveHook(trainer, save_name="NetSavedMultipleFont",
        save_after=1, overwrite=False))
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, test_loader,
                                                          vis_last_ep=False,
                                                          save=False,
                                                          req_shots_num=3))
    print("[*] Training")
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


if __name__ == '__main__':
    print(multifonts_evaluation(epochs=2))
