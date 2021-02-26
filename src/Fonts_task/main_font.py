# Writer: Gal Harari
# Date: 14/12/2020
from src.Fonts_task.font_prediction import FontDataLoader, MNISTDataLoader, IMG_SIZE
from src.brainNN import BrainNN
from src.hooks import ClassesEvalHook, OutputDistributionHook, SaveHook
from src.utils.train_utils import DefaultOptimizer, Trainer
from src.utils.general_utils import get_pparent_dir

import os

pparentdir = get_pparent_dir(__file__)

TRAIN_DIR = os.path.join(pparentdir, 'data/Font images/Calibri Font images/')
TEST_DIR = os.path.join(pparentdir, 'data/Font images/Calibri Font images/')


def create_trainer(data_loader, epochs=17):
    img_len = len(data_loader.samples[0])
    output_shape = len(data_loader.classes_neurons)

    fc = [BrainNN.FC]
    kernel = 3
    stride = 1
    into_n = 1
    rf = [BrainNN.RF, [kernel, stride, into_n]]

    nodes_details = [img_len, 144, output_shape * 2]
    IINs_details = [(4,), (4,), (4,)]
    conn_mat = [[fc, rf, None],
                [None, fc, None],
                [None, fc, fc]]
    img_dim = (IMG_SIZE, IMG_SIZE)
    spacial_dist_fac = 1.01
    iin_factor = 200
    into_iins_factor = 200
    vis_str = 'None'
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.CONNECTIONS_MAT: conn_mat,
                          BrainNN.SPACIAL_ARGS: img_dim,
                          BrainNN.SYNAPSE_SPACIAL_DISTANCE_FACTOR: spacial_dist_fac,
                          BrainNN.IINS_STRENGTH_FACTOR: iin_factor,
                          BrainNN.INTO_IINS_STRENGTH_FACTOR: into_iins_factor,
                          BrainNN.VISUALIZATION_FUNC_STR: vis_str}

    net = BrainNN(configuration_args)
    net.visualize_idle()
    optimizer = DefaultOptimizer(net=net, epochs=epochs, sample_reps=8, sharp=True,
                                 inc_prob=1, dec_prob=0.9)
    trainer = Trainer(net, data_loader, optimizer, verbose=True)
    return net, trainer


def fonts_trainer_evaluation(epochs=8):
    print("[*] Creating the trainer")
    data_loader = FontDataLoader(TRAIN_DIR, shuffle=True)
    net, trainer = create_trainer(data_loader, epochs)
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, FontDataLoader(
        TEST_DIR, batched=False), vis_last_ep=False))
    print("[*] Training")
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


def mnist_train_evaluate(epochs=8):
    print("[*] Creating the trainer")
    data_loader = MNISTDataLoader(idxs_lim=[0, 400], shuffle=True)
    net, trainer = create_trainer(data_loader, epochs)
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, MNISTDataLoader(
        idxs_lim=[0, 400], batched=False), vis_last_ep=False))
    trainer.register_hook(lambda trainer: SaveHook(trainer, save_after=8))
    print("[*] Training")
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


def mnist_output_dist(epochs=8):
    print("[*] Creating the trainer")
    data_loader = MNISTDataLoader(small=True, shuffle=True)
    net, trainer = create_trainer(data_loader, epochs)

    interest_label_neurons = [i for i in range(10)]
    trainer.register_hook(
        lambda trainer: OutputDistributionHook(trainer, MNISTDataLoader(
            small=True, batched=False), interest_label_neurons))

    print("[*] Training")
    trainer.train()
    cls_dist_str = OutputDistributionHook.CLS_DIST_STR
    return [[str(l), trainer.storage[cls_dist_str][l]] for l in interest_label_neurons]


if __name__ == '__main__':
    print(mnist_train_evaluate(epochs=7))
