# Writer: Gal Harari
# Date: 23/07/2020
from binary_prediction import create_binary_input_generator, N
from brainNN import BrainNN

if __name__ == '__main__':
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(6, 4), (1, 1), (1, 1)]
    inter_connections = [(True, False), (False, False), (True, False)]
    record_flags = [False, True]
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.VISUALIZATION_FUNC_STR: 'Nne',
                          BrainNN.SYNAPSES_INITIALIZE_MEAN: 5,
                          BrainNN.SHOOT_THRESHOLD: 40,
                          BrainNN.INTER_CONNECTIONS_PER_LAYER: inter_connections,
                          BrainNN.RECORD_FLAG: record_flags}

    # brainNN = BrainNN(configuration_args)
    brainNN = BrainNN.load_model()
    brainNN.train(create_binary_input_generator(inject_answer=False))
