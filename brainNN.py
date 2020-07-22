# Writer: Gal Harari
# Date: 15/07/2020
import numpy as np
import cv2
import os.path
import pickle

VISUALIZATION_WINDOW_NAME = 'BrainNN'
SAVE_NAME = 'saved_model'
SAVE_SUFFIX = '.npy'


class BrainNN:
    NODES_DETAILS = 'Layers details [population := (IO neurons num, preds neurons num)]'
    IINS_PER_LAYER_NUM = 'IINs per layer num [(IO IINs num, preds IINs num)]'
    SYNAPSES_INITIALIZE_MEAN = 'Initialization of synapses to this size'
    SYNAPSES_INITIALIZE_STD = 'The precentage of the INITIALIZE FACTOR that will be the' \
                              ' std'
    SYNAPSE_DISTANCE_FACTOR = 'The farther the population is, the connections get ' \
                              'weaker exponentially by this factor'
    IINS_STRENGTH_FACTOR = 'The IINs will stronger by this factor than the excitatory'
    SHOOT_THRESHOLD = 'Threshold that above the neuron will shoot'
    SYNAPSE_CHANGE_PROBABILITY = 'Probability for a synapse weight to change in ' \
                                 'case it provides the learning rule'
    MAX_WEIGHT_INCREASE = 'Max value the weights can increase by. If they are 0 they ' \
                          'get increased by this value'
    WEIGHTS_SUM_INTO_NEURON = 'The sum of weights that go into a single neuron.'
    VISUALIZATION_FUNC_STR = 'Visualization function to use'
    VISUALIZATION_SIZE = "Visualization image's size"
    VISUALIZATION_ARGS = 'Arguments for the visualization function'
    INTER_CONNECTIONS_PER_LAYER = 'Stats for each layer if it has inter-connections'

    default_configuration = {
        # Neuron parameters
        # [0][0] layer is the input layer. [-1][0] layer is the output layer
        NODES_DETAILS: [4, 10, 10],
        # This also sets the number of layers in population
        IINS_PER_LAYER_NUM: [(2, 1), (1, 3), (1, 1)],
        # In order for each population setup regarding the inter-connections to count,
        # it has to be the same number of layers in it.
        INTER_CONNECTIONS_PER_LAYER: [(False, True), (True, False), (True, True)],
        # This might be connected to " WEIGHTS_SUM_INTO_NEURON "
        SYNAPSES_INITIALIZE_MEAN: 100,
        SYNAPSES_INITIALIZE_STD: 0.05,
        # Should be lower than 1 if the synapses mean is lower than 1!
        SYNAPSE_DISTANCE_FACTOR: 2,
        IINS_STRENGTH_FACTOR: 5,
        SHOOT_THRESHOLD: 100,
        SYNAPSE_CHANGE_PROBABILITY: 0.5,
        MAX_WEIGHT_INCREASE: 50,
        # This might be connected to " SYNAPSES_INITIALIZE_MEAN "
        WEIGHTS_SUM_INTO_NEURON: 1000,
        VISUALIZATION_FUNC_STR: 'default',
        # [0] is width, [1] is height
        VISUALIZATION_SIZE: [700, 1500],
        VISUALIZATION_ARGS: None
    }


    def __init__(self, conf_args={}):
        self.__conf_args = {key: conf_args.get(key, BrainNN.default_configuration[key])
                            for key in BrainNN.default_configuration.keys()}

        self.__thresh = self.__conf_args[BrainNN.SHOOT_THRESHOLD]
        self.__change_prob = self.__conf_args[BrainNN.SYNAPSE_CHANGE_PROBABILITY]
        self.__max_weight_increase = self.__conf_args[BrainNN.MAX_WEIGHT_INCREASE]
        self.__weights_sum_into_neuron = self.__conf_args[BrainNN.WEIGHTS_SUM_INTO_NEURON]

        # Visualize section
        # Create visualization frame
        self.__visualize_dict = {
            'default': self.__default_visualize,
            'None': (lambda: None)
        }
        vis_size = self.__conf_args[BrainNN.VISUALIZATION_SIZE]
        self.__visualization_frame = np.zeros((vis_size[0], vis_size[1], 3))
        # Choose visualization function
        self.visualize = self.__visualize_dict[self.__conf_args[
            BrainNN.VISUALIZATION_FUNC_STR]]
        # Create visualization window
        self.__vis_window_name = VISUALIZATION_WINDOW_NAME
        cv2.namedWindow(self.__vis_window_name)
        # To make sure it's in the viewable screen
        cv2.moveWindow(self.__vis_window_name, 0, 0)

        # Initialize data structures
        self.__init_model()
        self.__init_sensory_input()


    def __init_model(self):
        excitatory_details, IINs_per_layer = self.__conf_args[BrainNN.NODES_DETAILS], \
                                             self.__conf_args[
                                                 BrainNN.IINS_PER_LAYER_NUM]
        assert len(excitatory_details) == len(IINs_per_layer), "Population number " \
                                                               "doesn't match between " \
                                                               "the layers details and " \
                                                               "the IINs details"

        # each sub-list is a population
        self.__neurons_per_layer = [[exitatories + IINs[i] for i in range(len(IINs))]
                                    for
                                    exitatories, IINs in list(zip(excitatory_details,
                                                                  IINs_per_layer))]

        self.__IINs_start_per_layer = excitatory_details[:]

        # each sub-list is a population
        self.__layers = BrainNN.__create_layers_np_arrays(self.__neurons_per_layer)

        # Required for iterating
        # This is what is added to the layers after each time step
        self.__change_in_layers = BrainNN.__create_layers_np_arrays(
            self.__neurons_per_layer)
        self.__prev_shots = BrainNN.__create_layers_np_arrays(self.__neurons_per_layer)
        self.__current_shots = BrainNN.__create_layers_np_arrays(self.__neurons_per_layer)

        self.__create_connections()


    @staticmethod
    def __create_layers_np_arrays(neurons_per_layer):
        return [[np.zeros(population_neurons[i]) for i in range(len(
            population_neurons))] for population_neurons in neurons_per_layer]


    def __create_connections(self):
        # Initialize the connections with some randomness
        mean = self.__conf_args[BrainNN.SYNAPSES_INITIALIZE_MEAN]
        std = self.__conf_args[BrainNN.SYNAPSES_INITIALIZE_STD] * mean

        # This determines for each layer if it will have inter-connections
        self.__inter_connections_flags = self.__conf_args[
            BrainNN.INTER_CONNECTIONS_PER_LAYER]

        # Arranged in [ population list[ layer list[matrices from layer to other
        # ones:= [matrix, (popul_dst_idx, layer_dst_idx) ] ] ]
        self.__synapses_matrices = []
        for popul_idx, population_neurons in enumerate(self.__neurons_per_layer):
            population_list = []
            self.__synapses_matrices.append(population_list)

            for cur_layer_idx, layer_neurons_num in enumerate(population_neurons):
                layer_list = []
                population_list.append(layer_list)

                # Now iterate over all layers and create connections. We will create
                # only forward connections
                for popul_idx_to_connect in range(popul_idx,
                                                  len(self.__neurons_per_layer)):
                    popul_to_connect = self.__neurons_per_layer[popul_idx_to_connect]

                    # Determine inter-connections inside the population's layers

                    popul_layers_inter_conns = self.__validate_inter_connections_format(
                        popul_idx, popul_to_connect)

                    for layer_to_conn_idx, layer_to_conn in enumerate(popul_to_connect):
                        # Create connections to layer "layer_to_connect" from current
                        # layer

                        # Don't create connections to different layers in different
                        # populations
                        if layer_to_conn_idx != cur_layer_idx and popul_idx_to_connect \
                                != popul_idx:
                            continue

                        idxs = tuple([popul_idx_to_connect, layer_to_conn_idx])
                        layer_list.append([np.random.normal(mean, std, (layer_neurons_num,
                                                                        layer_to_conn)),
                                           idxs])

                        # Prevent self loops
                        if layer_to_conn_idx == cur_layer_idx and popul_idx_to_connect \
                                == popul_idx:
                            # Here it means the layers are the same layer
                            # Check if it is defined to have inter-connections
                            # Default is to have inter-connections, so if it's not
                            # defined well it will have inter-connections
                            if popul_layers_inter_conns and not \
                                    popul_layers_inter_conns[ \
                                    cur_layer_idx]:
                                del layer_list[-1]
                                continue

                            # Prevent self loops
                            np.fill_diagonal(layer_list[-1][0], 0)
                        else:
                            # Here it means those are 2 different layers
                            # Prevent IINs of one layer to shoot to another
                            IINs_start_idx = self.__IINs_start_per_layer[popul_idx]
                            layer_list[-1][0][IINs_start_idx:, :] = 0
                            # Prevent excitatory neurons to shoot into other layer's IINs
                            IINs_start_idx_to_connect = self.__IINs_start_per_layer[
                                popul_idx_to_connect]
                            layer_list[-1][0][:, IINs_start_idx_to_connect:] = 0

                        # Weaken links between far populations, and strength inner
                        # population connections
                        layer_list[-1][0] *= (self.__conf_args[
                                                  BrainNN.SYNAPSE_DISTANCE_FACTOR] ** (
                                                      1 - abs(
                                                  popul_idx_to_connect -
                                                  popul_idx)))


    def __validate_inter_connections_format(self, popul_idx, popul_to_connect):
        popul_layers_inter_conns = self.__inter_connections_flags
        # Make sure the dimensions are right
        if len(popul_layers_inter_conns) != len(self.__layers) or \
                len(popul_layers_inter_conns[popul_idx]) != len(
            popul_to_connect):
            # It means the setup is not defined properly
            popul_layers_inter_conns = None
            print("The inter-connections flags are not the same dimension "
                  "as the layers in the model are!")
        else:
            # It means everything is fine
            popul_layers_inter_conns = self.__inter_connections_flags[
                popul_idx]
        return popul_layers_inter_conns


    def __init_sensory_input(self):
        # The sensory input array. Only to the lower layer in the first population
        self.__sensory_input = np.zeros(self.__neurons_per_layer[0][0])

        # Arrays to inject to the last population. To all layers in the last population
        last_popul = self.__layers[-1]
        self.__inject_to_last_popul = [np.zeros(last_popul[i].shape) for i
                                       in range(len(last_popul))]


    def save_state(self, name=None, overwrite=True):
        """
        Saves the synapses and the neurons. Later we can load only the connections or
        load the whole state
        :param name: name to save to. Without suffix.
        :param overwrite: set to False in case you wish to not overwrite previously
        saved models.
        :return:
        """
        save_name = name if name else SAVE_NAME

        # Don't overwrite previously saved models in case overwrite == False
        if not overwrite and os.path.isfile(save_name + SAVE_SUFFIX):
            i = 0
            while os.path.isfile(save_name + str(i) + SAVE_SUFFIX):
                i += 1
            save_name += str(i)

        with open(save_name + SAVE_SUFFIX, 'wb') as f:
            pickle.dump(self, f, 0)


    @staticmethod
    def load_model(name=None):
        """
        load the model. Default name to load from is the default name to save with.
        :param name: name to load from. Without suffix.
        :return:
        """
        load_name = name if name else SAVE_NAME
        with open(load_name + SAVE_SUFFIX, 'rb') as f:
            loaded = pickle.load(f)
        return loaded


    def train(self, input_generator):
        """
        :param input_generator: a functions that gets the model, update it's
        __sensory_input and __inject_to_last_popul. It returns False in case of finish
        or failure
        :return:
        """
        while input_generator(self):
            # Update the input and the injection to the last layer
            self.__layers[0][0] += self.__sensory_input
            for i, layer in enumerate(self.__layers[-1]):
                self.__layers[-1][i] += self.__inject_to_last_popul[i]

            # Iterate
            self.__iterate()

        self.save_state()


    def update_inject_and_input(self):
        """
        :return: True in case of success, False in case of failure or finish
        """
        self.visualize()


    def __iterate(self):
        """
        update all of the neurons and all of the connections one time step forward
        TODO: implement more efficient
        :return:
        """
        for cur_popul_idx, cur_popul in enumerate(self.__layers):
            for cur_layer_idx, cur_layer in enumerate(cur_popul):
                self.__current_shots[cur_popul_idx][cur_layer_idx] = (cur_layer >
                                                                      self.__thresh) * 1
                cur_shots = self.__current_shots[cur_popul_idx][cur_layer_idx]

                # Deduce from those who fired
                # Counting on that it resets after each time stamp
                self.__change_in_layers[cur_popul_idx][
                    cur_layer_idx] += cur_shots * cur_layer

                # IINs Deduce voltage
                INNs_idx = self.__IINs_start_per_layer[cur_popul_idx][cur_layer_idx]
                cur_shots[INNs_idx:] *= -1

                matrices_from_cur_layer = self.__synapses_matrices[cur_popul_idx][
                    cur_layer_idx]

                for matrix, dst_idxs in matrices_from_cur_layer:
                    self.__change_in_layers[dst_idxs[0]][
                        dst_idxs[1]] += cur_shots @ matrix

        # Update all layers and reset the "change_in_layers" help array
        for cur_popul_idx, cur_popul in enumerate(self.__layers):
            for cur_layer_idx, cur_layer in enumerate(cur_popul):
                cur_layer += self.__change_in_layers[cur_popul_idx][cur_layer_idx]
                self.__change_in_layers[cur_popul_idx][cur_layer_idx] *= 0

        self.__update_weights()


    def __update_weights(self):
        """
        TODO: implement more efficient. can be easily done by inserting it during the
              process of firing, as described in the doc.
        :return:
        """
        # Iterate over the layers
        for cur_popul_idx, cur_popul in enumerate(self.__layers):
            for cur_layer_idx in range(len(cur_popul)):
                # Get the shots of the current layer's neurons, in the previous time step
                cur_layer_prev_shots = self.__prev_shots[cur_popul_idx][cur_layer_idx]

                # Get the matrices out of the current layer to other layers
                matrices_from_cur_layer = self.__synapses_matrices[cur_popul_idx][
                    cur_layer_idx]

                for idx, mat_data in enumerate(matrices_from_cur_layer):
                    matrix, dst_idxs = mat_data
                    to_layer_cur_shots = self.__current_shots[dst_idxs[0]][dst_idxs[1]]

                    # who didn't shoot get -1 to decrease if one shots to them and they
                    # didn't shoot. Thats why ( 2 * to_layer_cur_shots - 1 ) to make 1
                    # to 1 and 0 to -1.
                    update_matrix = np.outer(cur_layer_prev_shots, 2 * to_layer_cur_shots
                                             - 1)
                    # I'm afraid it would be problematic so I add the assertion here
                    assert update_matrix.shape == (cur_layer_prev_shots.shape[0],
                                                   to_layer_cur_shots.shape[1])

                    self.__synapses_matrices[cur_popul_idx][cur_layer_idx][idx] = \
                        self.__update_synapses_matrix(update_matrix, matrix)


    def __update_synapses_matrix(self, shots_mat, cur_synapses_mat):
        # Update only on some probability. Otherwise leave the synapse unchanged by
        # setting the updation for it to zero
        shots_mat[shots_mat != 0] *= (
                np.random.uniform(
                    size=(np.count_nonzero(shots_mat))) < self.__change_prob)

        weighted_synapses_matrix = np.multiply(shots_mat, cur_synapses_mat)
        updated_mat = self.__create_update_mat(
            weighted_synapses_matrix) + cur_synapses_mat

        # Normalize the input synapses to each neuron
        return updated_mat / (updated_mat.sum(axis=0) / cur_synapses_mat.sum(axis=0))


    def __create_update_mat(self, weighted_synapses_matrix):
        return np.where(weighted_synapses_matrix > 0, 1 / (weighted_synapses_matrix +
                                                           (
                                                                   1 /
                                                                   self.__max_weight_increase)),
                        weighted_synapses_matrix / 2)


    def set_visualization(self, vis_func_str):
        self.visualize = self.__visualize_dict.get(vis_func_str,
                                                   self.__default_visualize)


    def __default_visualize(self, debug=False):
        """
        visualize the model according to it's current state and the visualization
        parameters chosen in the beginning.
        :return:
        """
        # TODO: DEBUG THIS FUNCTION! no way it will work for the first time
        frame = self.__visualization_frame
        neuron_size = 5
        line_thick = 1

        # Decrease the neuron size to keep the avoid cutting of the neurons
        h, w, _ = frame.shape
        w, h = w - 2 * neuron_size, h - 2 * neuron_size

        popul_gap = w // len(self.__layers)
        for popul_idx, population_neurons in enumerate(self.__layers):
            layers_num = len(population_neurons)
            layer_gap_y = h // layers_num
            layer_gap_x = popul_gap // layers_num

            for cur_layer_idx, cur_layer in enumerate(population_neurons):
                neurons_gap = layer_gap_y // (len(cur_layer) + 1)

                for neuron_idx, neuron in enumerate(cur_layer):
                    location = (neuron_size + popul_idx * popul_gap + cur_layer_idx *
                                layer_gap_x, neuron_size + cur_layer_idx * layer_gap_y +
                                neuron_idx * neurons_gap)

                    # First draw connections to all other neurons
                    synapse_matrices = self.__synapses_matrices[popul_idx][cur_layer_idx]

                    for mat, idxs in synapse_matrices:
                        to_popul_idx, to_layer_idx = idxs
                        to_population_neurons = self.__layers[to_popul_idx]

                        layers_num = len(to_population_neurons)
                        to_layer_gap_y = h // layers_num
                        to_layer_gap_x = popul_gap // layers_num
                        to_layer = to_population_neurons[to_layer_idx]
                        to_neurons_gap = to_layer_gap_y // (len(to_layer) + 1)

                        for to_neuron_idx, to_neuron in enumerate(to_layer):
                            to_location = (neuron_size + to_popul_idx * popul_gap +
                                           to_layer_idx * to_layer_gap_x, neuron_size +
                                           to_layer_idx * to_layer_gap_y + to_neuron_idx
                                           * to_neurons_gap)
                            # First draw the connections
                            connections_strength = mat[neuron_idx, to_neuron_idx]
                            if connections_strength:
                                cv2.line(frame, location, to_location,
                                         (0, connections_strength, 0),
                                         line_thick)

                    # now draw the remaining neurons
                    cv2.circle(frame, location, neuron_size,
                               (int(neuron), int(neuron), int(neuron)),
                               thickness=-1)
                    # Show the IINs by drawing a circle around them
                    if neuron_idx >= self.__IINs_start_per_layer[popul_idx]:
                        cv2.circle(frame, location, neuron_size, (0, 0, 255), thickness=1)

                    # In case of debugging, see each step
                    if debug:
                        cv2.imshow(self.__vis_window_name, frame)
                        cv2.waitKey(0)

        cv2.imshow(self.__vis_window_name, frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    brainNNmodel = BrainNN()
    brainNNmodel.visualize()
    cv2.waitKey()
