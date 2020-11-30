# Writer: Gal Harari
# Date: 23/07/2020
from binary_prediction import create_binary_input_generator, N, \
    evaluate_binary_representation_nn, BinaryDataLoader
from brainNN import BrainNN
from utils.train_utils import Trainer, DefaultOptimizer, TrainNetWrapper
import numpy as np
from hooks import ClassesEvalHook

LOAD = False


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
                          BrainNN.INTER_CONNECTIONS_PER_LAYER: inter_connections,
                          BrainNN.RECORD_FLAG: record_flags,
                          BrainNN.IINS_STRENGTH_FACTOR: 2,
                          BrainNN.SYNAPSE_INCREASE_FUNC: increase_func,
                          BrainNN.SYNAPSE_DECREASE_FUNC: decrease_func,
                          BrainNN.SYNAPSE_DECREASE_PROBABILITY: dec_prob,
                          BrainNN.SYNAPSE_INCREASE_PROBABILITY: inc_prob}

    if LOAD:
        brainNN = BrainNN.load_model()
    else:
        brainNN = BrainNN(configuration_args)

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


def trainer_evaluation():
    net, trainer = create_trainer()
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, BinaryDataLoader()))
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str]]


def create_trainer(epoches=17):
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(3, 3), (3, 3), (1, 1)]
    inter_connections = [(False, True), (True, True), (True, True)]
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.INTER_CONNECTIONS_PER_LAYER: inter_connections}

    net = BrainNN(configuration_args)
    data_loader = BinaryDataLoader(shuffle=True)
    optimizer = DefaultOptimizer(net=net, epoches=epoches, sample_reps=6)
    trainer = Trainer(net, data_loader, optimizer, verbose=False)
    return net, trainer


if __name__ == '__main__':
    # script_training(epoches=10)
    trainer_train(epoches=10)
