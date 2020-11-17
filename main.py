# Writer: Gal Harari
# Date: 23/07/2020
from binary_prediction import create_binary_input_generator, N, \
    evaluate_binary_representation_nn
from brainNN import BrainNN

LOAD = False

if __name__ == '__main__':
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(3, 3), (3, 3), (1, 1)]
    inter_connections = [(False, True), (True, True), (True, True)]
    record_flags = [True, True]
    increase_func = lambda weights: 0.1,
    decrease_func = lambda neg_weights: 0.1
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
                          BrainNN.IINS_STRENGTH_FACTOR: 2}

    if LOAD:
        brainNN = BrainNN.load_model()
    else:
        brainNN = BrainNN(configuration_args)

    brainNN.set_visualization(vis_str)

    brainNN.train(create_binary_input_generator(inject_answer=True, epoches=10,
                                                verbose=False))

    brainNN.set_visualization(eval_vis_str)
    evaluate_binary_representation_nn(brainNN)
