# Writer: Gal Harari
# Date: 14/12/2020
from src.Fonts_task.font_prediction import FontDataLoader, IMG_SIZE
from src.brainNN import BrainNN
from src.hooks import ClassesEvalHook
from src.utils.train_utils import DefaultOptimizer, Trainer
from src.utils.general_utils import get_pparent_dir

import os

pparentdir = get_pparent_dir(__file__)

TRAIN_DIR = os.path.join(pparentdir, 'data/Font images/Calibri Font images/')
TEST_DIR = os.path.join(pparentdir, 'data/Font images/Calibri Font images/')


def create_trainer(epoches=17):
    data_loader = FontDataLoader(TRAIN_DIR, shuffle=True)
    img_len = len(data_loader.samples[0])
    output_shape = len(data_loader.classes_neurons)

    fc = [BrainNN.FC]
    kernel = 3
    stride = 1
    rf = [BrainNN.RF, [kernel, stride]]

    nodes_details = [img_len, 144, output_shape]
    IINs_details = [(4,), (4,), (4,)]
    conn_mat = [[fc, rf, None],
                [None, fc, fc],
                [None, None, fc]]
    img_dim = (IMG_SIZE, IMG_SIZE)
    spacial_dist_fac = 1.01
    iin_factor = 20
    into_iins_factor = 1
    vis_str = 'None'
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.CONNECTIONS_MAT: conn_mat,
                          BrainNN.SPACIAL_ARGS: img_dim,
                          BrainNN.SYNAPSE_SPACIAL_DISTANCE_FACTOR: spacial_dist_fac,
                          # BrainNN.IINS_STRENGTH_FACTOR: iin_factor,
                          BrainNN.INTO_IINS_STRENGTH_FACTOR: into_iins_factor,
                          BrainNN.VISUALIZATION_FUNC_STR: vis_str}

    net = BrainNN(configuration_args)
    net.visualize_idle()
    optimizer = DefaultOptimizer(net=net, epochs=epoches, sample_reps=8, sharp=True,
                                 inc_prob=1, dec_prob=0.9)
    trainer = Trainer(net, data_loader, optimizer, verbose=False)
    return net, trainer


def fonts_trainer_evaluation(epoches=10):
    print("[*] Creating the trainer")
    net, trainer = create_trainer(epoches)
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, FontDataLoader(
        TEST_DIR, batched=True), vis_last_ep=False))
    print("[*] Training")
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


if __name__ == '__main__':
    print(fonts_trainer_evaluation(epoches=7))
