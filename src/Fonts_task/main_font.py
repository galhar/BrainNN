# Writer: Gal Harari
# Date: 14/12/2020
from src.Fonts_task.font_prediction import FontDataLoader, MNISTDataLoader, IMG_SIZE
from src.brainNN import BrainNN, SAVE_NAME, SAVE_SUFFIX
from src.hooks import ClassesEvalHook, OutputDistributionHook, SaveHook, SaveByEvalHook
from src.utils.train_utils import DefaultOptimizer, Trainer
from src.utils.general_utils import get_pparent_dir

import os

pparentdir = get_pparent_dir(__file__)

TRAIN_DIR = os.path.join(pparentdir, 'data/Font images/Calibri Font images/')
TEST_DIR = os.path.join(pparentdir, 'data/Font images/Calibri Font images/')

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
    net.visualize_idle()
    optimizer = DefaultOptimizer(net=net, epochs=epochs, sample_reps=11, sharp=True,
                                 inc_prob=0.9, dec_prob=0.0)
    trainer = Trainer(net, data_loader, optimizer, verbose=True)
    return net, trainer


def fonts_trainer_evaluation(epochs=20):
    print("[*] Creating the trainer")
    data_loader = FontDataLoader(TRAIN_DIR, shuffle=True)
    net, trainer = create_trainer(data_loader, epochs)
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, FontDataLoader(
        TEST_DIR, batched=False, noise_std=2 * 1/4), vis_last_ep=False, save=True))
    trainer.register_hook(lambda trainer: SaveByEvalHook(trainer, req_acc=70))
    trainer.register_hook(
        lambda trainer: SaveHook(trainer, save_after=1, overwrite=False))
    print("[*] Training")
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


def mnist_train_evaluate(epochs=40):
    print("[*] Creating the trainer")
    data_loader = MNISTDataLoader(small=True, shuffle=True, amp=0.03)
    net, trainer = create_trainer(data_loader, epochs)
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, MNISTDataLoader(
        small=True, batched=False, amp=0.03), vis_last_ep=False, save=True))
    trainer.register_hook(
        lambda trainer: SaveHook(trainer, save_after=1, overwrite=False))
    print("[*] Training")
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


def mnist_output_dist(epochs=8):
    print("[*] Creating the trainer")
    data_loader = MNISTDataLoader(small=True, shuffle=True, amp=0.03)
    net, trainer = create_trainer(data_loader, epochs)

    interest_label_neurons = [i for i in range(10)]
    trainer.register_hook(
        lambda trainer: OutputDistributionHook(trainer, MNISTDataLoader(
            small=True, batched=False, amp=0.03), interest_label_neurons))

    print("[*] Training")
    trainer.train()
    cls_dist_str = OutputDistributionHook.CLS_DIST_STR
    return [[str(l), trainer.storage[cls_dist_str][l]] for l in interest_label_neurons]


if __name__ == '__main__':
    print(mnist_train_evaluate())
