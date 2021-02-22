# Writer: Gal Harari
# Date: 23/07/2020
from src.binary_encoding_task.binary_prediction import create_binary_input_generator, N, \
    evaluate_binary_representation_nn, BinaryDataLoader, BinaryOneOneDataLoader, \
    BinaryOneOneEvalHook
from src.brainNN import BrainNN
# from src.BrainNN_prev import BrainNN
from src.utils.train_utils import Trainer, DefaultOptimizer
import numpy as np
from src.hooks import ClassesEvalHook, SaveByEvalHook, OutputDistributionHook
import cv2
from deprecated import deprecated

LOAD = False
SAVE_NAME = 'prev_version_save.json'


@deprecated(reason="This method isn't supported by the 'Trainer' hierarchy")
def script_training(epoches=14):
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(3, 3), (3, 3), (1, 1)]
    inter_connections = [(False, True), (True, True), (True, True)]
    record_flags = [True, True]
    increase_func = lambda weights: np.full(weights.shape, 0.1)
    decrease_func = lambda neg_weights: np.maximum(neg_weights / 2, -0.04)
    inc_prob, dec_prob = 0.7, 0.2
    vis_str = 'None'
    eval_vis_str = 'None'
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.VISUALIZATION_FUNC_STR: vis_str,
                          BrainNN.SYNAPSES_INITIALIZE_MEAN: 5,
                          BrainNN.SHOOT_THRESHOLD: 40,
                          BrainNN.CONNECTIONS_MAT: inter_connections,
                          BrainNN.RECORD_FLAG: record_flags,
                          BrainNN.IINS_STRENGTH_FACTOR: 2,
                          BrainNN.SYNAPSE_INCREASE_FUNC: increase_func,
                          BrainNN.SYNAPSE_DECREASE_FUNC: decrease_func,
                          BrainNN.SYNAPSE_DECREASE_PROBABILITY: dec_prob,
                          BrainNN.SYNAPSE_INCREASE_PROBABILITY: inc_prob}

    if LOAD:
        brainNN = BrainNN.load_model({BrainNN.VISUALIZATION_FUNC_STR: 'default'})
    else:
        brainNN = BrainNN(configuration_args)

    brainNN.visualize()
    cv2.waitKey()

    brainNN.set_visualization(vis_str)

    brainNN.train(create_binary_input_generator(inject_answer=True, epoches=epoches,
                                                repeat_sample=6, verbose=False))

    brainNN.set_visualization(eval_vis_str)
    brainNN.freeze()
    return evaluate_binary_representation_nn(brainNN, noise=0, req_shots=5)


def trainer_train(epoches=1):
    net, trainer = create_trainer(epoches)
    trainer.train()

    net.freeze()
    return evaluate_binary_representation_nn(net, noise=0, req_shots=5)


def trainer_evaluation(epoches=10):
    net, trainer = create_trainer(epoches)
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, BinaryDataLoader(
        batched=True)))
    trainer.register_hook(lambda trainer: SaveByEvalHook(trainer, req_acc=70))
    trainer.train()
    net.plot_history()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


def output_distribution_query(epoches=8):
    net, trainer = create_trainer(epoches)
    interest_label_neurons = [i for i in range(10)]
    trainer.register_hook(
        lambda trainer: OutputDistributionHook(trainer, BinaryDataLoader(
            batched=True), interest_label_neurons))
    trainer.train()
    cls_dist_str = OutputDistributionHook.CLS_DIST_STR
    return [[str(l), trainer.storage[cls_dist_str][l]] for l in interest_label_neurons]


def create_trainer(epoches=17):
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(3, 3), (3, 3), (1, 1)]
    conn_mat = [[[BrainNN.FC], [BrainNN.FC], None],
                [None, [BrainNN.FC], [BrainNN.FC]],
                [None, [BrainNN.FC], [BrainNN.FC]]]
    iins_factor = 20
    into_iins_factor = 20
    vis_str = 'None'
    configuration_args = {
        BrainNN.NODES_DETAILS: nodes_details,
        BrainNN.IINS_PER_LAYER_NUM: IINs_details,
        BrainNN.CONNECTIONS_MAT: conn_mat,
        BrainNN.IINS_STRENGTH_FACTOR: iins_factor,
        BrainNN.INTO_IINS_STRENGTH_FACTOR: into_iins_factor,
        BrainNN.VISUALIZATION_FUNC_STR: vis_str
    }

    if LOAD:
        net = BrainNN.load_model(name=SAVE_NAME)
    else:
        net = BrainNN(configuration_args)
    data_loader = BinaryDataLoader(shuffle=True)
    optimizer = DefaultOptimizer(net=net, epochs=epoches, sample_reps=6, sharp=1,
                                 inc_prob=1, dec_prob=0.8)
    trainer = Trainer(net, data_loader, optimizer, verbose=False)
    return net, trainer


def one_one_evaluation(epochs=16):
    # Create trainer:
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(3, 3), (3, 3), (1, 1)]
    inter_connections = [(False, True), (True, True), (True, True)]
    feedback = False
    iins_factor = 10
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.CONNECTIONS_MAT: inter_connections,
                          BrainNN.FEEDBACK: feedback,
                          BrainNN.IINS_STRENGTH_FACTOR: iins_factor}

    net = BrainNN(configuration_args)
    data_loader = BinaryDataLoader(shuffle=True)
    optimizer = DefaultOptimizer(net=net, epochs=epochs, sample_reps=6)
    trainer = Trainer(net, data_loader, optimizer, verbose=False)

    trainer.register_hook(lambda trainer: BinaryOneOneEvalHook(trainer, BinaryDataLoader(
        batched=True)))
    trainer.register_hook(lambda trainer: SaveByEvalHook(trainer, req_acc=70))
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


if __name__ == '__main__':
    # script_training(epoches=1)
    # trainer_train(epoches=3)
    print(trainer_evaluation(epoches=1))
    # print(one_one_evaluation())
