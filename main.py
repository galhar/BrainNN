# Writer: Gal Harari
# Date: 23/07/2020
from binary_prediction import create_binary_input_generator, N, \
    evaluate_binary_representation_nn
from brainNN import BrainNN

LOAD = False

if __name__ == '__main__':
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(3, 3), (3, 2), (1, 1)]
    inter_connections = [(False, True), (True, True), (True, True)]
    record_flags = [True, True]
    vis_str = 'None'
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.VISUALIZATION_FUNC_STR: vis_str,
                          BrainNN.SYNAPSES_INITIALIZE_MEAN: 5,
                          BrainNN.SHOOT_THRESHOLD: 40,
                          BrainNN.INTER_CONNECTIONS_PER_LAYER: inter_connections,
                          BrainNN.RECORD_FLAG: record_flags}

    if LOAD:
        brainNN = BrainNN.load_model()
    else:
        brainNN = BrainNN(configuration_args)

    brainNN.set_visualization(vis_str)

    brainNN.train(create_binary_input_generator(inject_answer=True, cycles=1))

    evaluate_binary_representation_nn(brainNN)
