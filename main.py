# Writer: Gal Harari
# Date: 23/07/2020
from binary_prediction import create_binary_input_generator, N, \
    evaluate_binary_representation_nn
from brainNN import BrainNN
import numpy as np

LOAD = False

if __name__ == '__main__':
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(3, 3), (3, 3), (1, 1)]
    inter_connections = [(False, True), (True, True), (True, True)]
    record_flags = [True, True]
    increase_func = lambda weights: np.full(weights.shape, 0.1)
    decrease_func = lambda neg_weights: np.full( neg_weights.shape, -0.1)
    inc_prob, dec_prob = 0.7, 0.5
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
                          # BrainNN.SYNAPSE_INCREASE_FUNC: increase_func,
                          # BrainNN.SYNAPSE_DECREASE_FUNC: decrease_func,
                          BrainNN.SYNAPSE_DECREASE_PROBABILITY: dec_prob,
                          BrainNN.SYNAPSE_INCREASE_PROBABILITY: inc_prob}

    if LOAD:
        brainNN = BrainNN.load_model()
    else:
        brainNN = BrainNN(configuration_args)

    brainNN.set_visualization(vis_str)

    brainNN.train(create_binary_input_generator(inject_answer=True, epoches=1,
                                                verbose=False))

    brainNN.set_visualization(eval_vis_str)
    evaluate_binary_representation_nn(brainNN)
