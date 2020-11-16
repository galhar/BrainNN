# Writer: Gal Harari
# Date: 15/07/2020
import numpy as np
import cv2
import os.path
import dill

VISUALIZATION_WINDOW_NAME = 'BrainNN'
SAVE_NAME = 'saved_model'
SAVE_SUFFIX = '.pkl'
RECORD_SAVE_NAME = 'visualization_video'


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
    SYNAPSE_INCREASE_PROBABILITY = 'Probability for a synapse weight to INCREASE in ' \
                                   'case it provides the learning rule'
    SYNAPSE_DECREASE_PROBABILITY = 'Probability for a synapse weight to DECREASE in ' \
                                   'case it provides the learning rule'
    SYNAPSE_MEMORY_FACTOR = 'The synapses are updated according to shots into them in ' \
                            'previous steps. Thus the previous shots into a layer is a ' \
                            'moving average of the shots into this layer. Each time ' \
                            'step advancing, the previous average is multiplied by THIS ' \
                            '' \
                            '' \
                            '' \
                            '' \
                            '' \
                            '' \
                            '' \
                            '' \
                            'FACTOR and added to the current shots mulitplied by (' \
                            '1-THIS FACTOR). RANGE: [0,1]. higher than 0.5 will cause ' \
                            'it to remember previous shots, lower will forget prev shots'
    MAX_WEIGHT_INCREASE = 'Max value the weights can increase by. If they are 0 they ' \
                          'get increased by this value'
    WEIGHTS_SUM_INTO_NEURON = 'The sum of weights that go into a single neuron.'
    VISUALIZATION_FUNC_STR = 'Visualization function to use'
    VISUALIZATION_SIZE = "Visualization image's size"
    VISUALIZATION_ARGS = 'Arguments for the visualization function'
    INTER_CONNECTIONS_PER_LAYER = 'Stats for each layer if it has inter-connections'
    MAX_CONNECTIONS_RES_IN_VISUALIZATION = 'Any value above this value will be seen as ' \
                                           'it is equal to this value in the ' \
                                           'visualization using colors'
    RECORD_FLAG = 'Record the chosen visualization of the run. Also allows to disable ' \
                  'viewing if recording. [to_record_, to_show_during_run]'

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
        SYNAPSES_INITIALIZE_MEAN: 8,
        SYNAPSES_INITIALIZE_STD: 0.15,
        # Should be lower than 1 if the synapses mean is lower than 1!
        SYNAPSE_DISTANCE_FACTOR: 3,
        IINS_STRENGTH_FACTOR: 2,
        SHOOT_THRESHOLD: 100,
        SYNAPSE_INCREASE_PROBABILITY: 0.8,
        SYNAPSE_DECREASE_PROBABILITY: 0.7,
        SYNAPSE_MEMORY_FACTOR: 0.6,
        MAX_WEIGHT_INCREASE: 8,
        # This might be connected to " SYNAPSES_INITIALIZE_MEAN "
        WEIGHTS_SUM_INTO_NEURON: 1000,
        VISUALIZATION_FUNC_STR: 'default',
        # [0] is width, [1] is height
        VISUALIZATION_SIZE: [700, 1300],
        MAX_CONNECTIONS_RES_IN_VISUALIZATION: 256,
        VISUALIZATION_ARGS: None,
        RECORD_FLAG: [False, True]

    }


    def __init__(self, conf_args={}):
        self.__conf_args = {key: conf_args.get(key, BrainNN.default_configuration[key])
                            for key in BrainNN.default_configuration.keys()}

        self.__thresh = self.__conf_args[BrainNN.SHOOT_THRESHOLD]
        self.__change_prob_increase = self.__conf_args[
            BrainNN.SYNAPSE_INCREASE_PROBABILITY]
        self.__change_prob_decrease = self.__conf_args[
            BrainNN.SYNAPSE_DECREASE_PROBABILITY]
        self.__synapse_memory_factor = self.__conf_args[BrainNN.SYNAPSE_MEMORY_FACTOR]
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
        h, w, _ = self.__visualization_frame.shape
        # Choose visualization function
        self.set_visualization(self.__conf_args[BrainNN.VISUALIZATION_FUNC_STR])

        # Create visualization window
        self.__vis_window_name = VISUALIZATION_WINDOW_NAME
        cv2.namedWindow(self.__vis_window_name)
        # To make sure it's in the viewable screen
        cv2.moveWindow(self.__vis_window_name, 0, 0)
        self.__vis_record = self.__conf_args[BrainNN.RECORD_FLAG]
        assert len(self.__vis_record) == 2, "RECORD_FLAG format must be [to_record, " \
                                            "to_show_during_run]"
        self.__record_writer = None
        if self.__vis_record[0]:
            self.__record_writer = cv2.VideoWriter(RECORD_SAVE_NAME + '.avi',
                                                   cv2.VideoWriter_fourcc(*'XVID'),
                                                   10, (w, h))

        # Initialize data structures
        self.__sensory_input = None
        self.__inject_to_last_popul = None
        self.__init_model()
        self.__init_sensory_input()

        # Those should be the max expected value of the connections and of the neurons.
        # It's the max value from which every higher value will be with the same color
        # in the visualization
        self.__connections_max_res = self.__determine_weights_res()
        self.__neurons_max_res = self.__thresh


    def __determine_weights_res(self):
        max_value = 0.
        # Iterate only on synapses from the first layer, assuming it represents well
        # all the others
        for mat, idx in self.__synapses_matrices[0][0]:
            mat_max = np.max(mat)
            max_value = mat_max if mat_max > max_value else max_value

        # That's not accurate! It's not that a weights can only increase by "max weight
        # increase", that's the max it will increase on 1 updating. But that's the
        # approximation
        return max_value + self.__max_weight_increase


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

        self.__IINs_start_per_popul = excitatory_details[:]

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
        IINs_factor = self.__conf_args[BrainNN.IINS_STRENGTH_FACTOR]

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

                # Normalize each neuron's output. It depends only on the weights in the
                # matrices from its layer to others, only on its relevant row on these
                # matrices. So normalize the values in each row over the matrices.
                normalize_vec = np.zeros((layer_neurons_num,))
                for popul_idx_to_connect in range(popul_idx,
                                                  len(self.__neurons_per_layer)):
                    popul_to_connect = self.__neurons_per_layer[popul_idx_to_connect]

                    # Determine inter-connections inside the population's layers

                    popul_layers_inter_conns = self.__validate_inter_connections_format(
                        popul_idx, popul_to_connect)

                    for layer_to_conn_idx, layer_to_conn in enumerate(popul_to_connect):
                        # Create connections to layer "layer_to_connect" from current
                        # layer
                        self.__create_connections_between_2_layers(layer_neurons_num,
                                                                   layer_to_conn,
                                                                   layer_list,
                                                                   cur_layer_idx,
                                                                   layer_to_conn_idx,
                                                                   popul_idx,
                                                                   popul_idx_to_connect,
                                                                   mean, std, IINs_factor,
                                                                   popul_layers_inter_conns)
                        normalize_vec += layer_list[-1][0].sum(axis=1)

                # Now normalize the output from each neuron as explained above the loop
                for i in range(len(layer_list)):
                    mat = layer_list[i][0]
                    layer_list[i][0] = self.__thresh * mat / normalize_vec[:, np.newaxis]


    def __create_connections_between_2_layers(self, layer_neurons_num,
                                              l_to_conn_neurons_n,
                                              layer_list,
                                              cur_layer_idx,
                                              layer_to_conn_idx, popul_idx,
                                              popul_idx_to_connect, mean, std,
                                              IINs_factor,
                                              popul_layers_inter_conns):
        # Don't create connections to different layers in different
        # populations
        if layer_to_conn_idx != cur_layer_idx and popul_idx_to_connect \
                != popul_idx:
            return

        idxs = tuple([popul_idx_to_connect, layer_to_conn_idx])
        layer_list.append([np.random.normal(mean, std, (layer_neurons_num,
                                                        l_to_conn_neurons_n)),
                           idxs])

        IINs_start_idx = self.__IINs_start_per_popul[popul_idx]

        # make the IINs stronger by the factor in the setup
        layer_list[-1][0][IINs_start_idx:, :] *= IINs_factor

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
                # Here it is set to not have inter-connections. The IINs are the only
                # neurons to connect to the others
                layer_list[-1][0][:IINs_start_idx, :IINs_start_idx] = 0
                # del layer_list[-1]
                return

            # layer_list[-1][0] is the matrix. layer_list[-1][1] are the idxs
            # Prevent self loops.
            np.fill_diagonal(layer_list[-1][0], 0)
        else:
            # Here it means those are 2 different layers

            # In the ame population create only between nodes
            if popul_idx_to_connect == popul_idx:
                inner_non_IINS_mat = layer_list[-1][0][:IINs_start_idx, :IINs_start_idx]
                idxs_to_delete_in_non_IIN_part = ~np.eye(inner_non_IINS_mat.shape[0],
                                                         dtype=bool)
                # Delete all but diagonal in the non IINs part of the matrix
                layer_list[-1][0][:IINs_start_idx, :IINs_start_idx][
                    idxs_to_delete_in_non_IIN_part] = 0

            # Prevent IINs of one layer to shoot to another
            layer_list[-1][0][IINs_start_idx:, :] = 0

            # # Prevent excitatory neurons to shoot into other layer's IINs
            # IINs_start_idx_to_connect = self.__IINs_start_per_popul[
            #     popul_idx_to_connect]
            # layer_list[-1][0][:, IINs_start_idx_to_connect:] = 0

        # Weaken links between far populations, and strength inner
        # population connections
        dist_fac = self.__conf_args[BrainNN.SYNAPSE_DISTANCE_FACTOR]

        # for the same layer treat differently:
        if popul_idx_to_connect == popul_idx:
            # connections in the same layer get weaken as the neurons are far from each
            # other, and node connections get stronger

            # only mess with the excitatory neurons
            extory_num = self.__IINs_start_per_popul[popul_idx]

            size = layer_list[-1][0].shape
            mult_mat = np.ones(size)
            for i in range(size[0]):
                for j in range(size[1]):
                    if i < extory_num and j < extory_num:
                        mult_mat[i, j] *= dist_fac ** (1-abs(i - j))

            layer_list[-1][0] *= mult_mat
        else:
            # connections between layers
            layer_list[-1][0] *= (dist_fac ** (1 - abs(popul_idx_to_connect - popul_idx)))


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
        # The sensory input array. Only to the lower layer in the first population,
        # not to the IINs in it
        self.__sensory_input = np.zeros(self.__IINs_start_per_popul[0])

        # Arrays to inject to the last population. To all layers in the last population
        last_popul_excitatory_neurons_num = self.__IINs_start_per_popul[-1]
        self.__inject_to_last_popul = [
            np.zeros(last_popul_excitatory_neurons_num) for i
            in range(len(self.__layers[-1]))]


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
            dill.dump(self, f)


    @staticmethod
    def load_model(name=None):
        """
        load the model. Default name to load from is the default name to save with.
        :param name: name to load from. Without suffix.
        :return:
        """
        load_name = name if name else SAVE_NAME
        with open(load_name + SAVE_SUFFIX, 'rb') as f:
            loaded = dill.load(f)
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
            IINs_start_inp_popul = self.__IINs_start_per_popul[0]
            IINs_start_out_popul = self.__IINs_start_per_popul[-1]
            self.__layers[0][0][:IINs_start_inp_popul] += self.__sensory_input
            for i, layer in enumerate(self.__layers[-1]):
                self.__layers[-1][i][:IINs_start_out_popul] += \
                    self.__inject_to_last_popul[i]

            # Iterate
            self.__iterate()
            self.visualize()

        if self.__record_writer:
            self.__record_writer.release()
        self.save_state()


    def update_inject_and_input(self):
        """
        :return: True in case of success, False in case of failure or finish
        """
        self.visualize()


    def zero_neurons(self):
        for popul in self.__layers:
            for layer in popul:
                layer.fill(0)


    def set_sensory_input(self, arr):
        assert arr.shape == self.__sensory_input.shape, "Sensory input inserted is in " \
                                                        "wrong shape!"
        self.__sensory_input = arr


    def set_last_popul_injection(self, arr_list):
        """
        Sets the input to inject to the last population during training
        :param arr_list: Must be in the same shape as the last population in the
        brainNN. None will result in no change in the arrays. None for all the arrayws
        is possible, or a list with None for the array we do not wish to change
        :return:
        """
        # For None
        if arr_list is None:
            return

        # Make sure all the dimension are right
        if len(arr_list) == len(self.__inject_to_last_popul):
            for inject_arr_idx, inject_arr in enumerate(arr_list):
                # Allow not to inject to a specific layer
                if inject_arr is None:
                    continue
                if inject_arr.shape != self.__inject_to_last_popul[inject_arr_idx].shape:
                    raise ValueError("inject to last popult during training with wrong "
                                     "dimension!")
                self.__inject_to_last_popul[inject_arr_idx] = arr_list[inject_arr_idx]


    def get_shot_threshold(self):
        return self.__thresh


    def get_output(self, get_shots=False, whole_popul=False):
        """
        Return the output layers, only the excitatory neurons in it.
        :param whole_popul: if set to True, the whole last population is returned. In
        this case also the IINs in the last populations will be included
        :param get_shots: if set to True, the shots instead of the values will be returned
        :return: The output of the brainNN
        """
        last_popul = self.__current_shots[-1] if get_shots else self.__layers[-1]
        excitatory_neurons_num = self.__IINs_start_per_popul[-1]
        if whole_popul:
            return last_popul
        return last_popul[0][:excitatory_neurons_num]


    def get_sensory_input(self):
        """
        Only for debug purposes
        :return:
        """
        return self.__sensory_input


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
                    cur_layer_idx] -= cur_shots * cur_layer

                # IINs Deduce voltage
                INNs_idx = self.__IINs_start_per_popul[cur_popul_idx]
                cur_shots[INNs_idx:] *= -1

                matrices_from_cur_layer = self.__synapses_matrices[cur_popul_idx][
                    cur_layer_idx]

                for matrix, dst_idxs in matrices_from_cur_layer:
                    self.__change_in_layers[dst_idxs[0]][
                        dst_idxs[1]] += cur_shots @ matrix

        # Update all layers and reset the "change_in_layers" help array
        for cur_popul_idx, cur_popul in enumerate(self.__layers):
            for cur_layer_idx, cur_layer in enumerate(cur_popul):
                # Make sure the IINs doesn't diminish too much to negatate the value
                cur_layer += np.maximum(self.__change_in_layers[cur_popul_idx][
                                            cur_layer_idx], -cur_layer)
                self.__change_in_layers[cur_popul_idx][cur_layer_idx] *= 0

        self.__update_weights()


    def __update_weights(self):
        """
        TODO: implement more efficient. can be easily done by inserting it during the
              process of firing, as described in the doc.
        :return:
        """
        syn_mem_fac = self.__synapse_memory_factor
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
                    dst_layer_cur_shots = self.__current_shots[dst_idxs[0]][dst_idxs[1]]

                    # who didn't previously shot get -1 to decrease if the dst neuron
                    # shot anyway. That's why ( 2 * cur_layer_prev_shots - 1 ) to make 1
                    # to 1 and 0 to -1. The cur_layer_prev_shots is a moving average of
                    # previous shots. Above 0.5 will count as it shot, bellow and equal
                    # will count as it didn't shot.
                    shots_matrix = np.outer(2 * (cur_layer_prev_shots > 0.5) - 1,
                                            dst_layer_cur_shots)
                    # I'm afraid it would be problematic so I add the assertion here
                    assert shots_matrix.shape == (cur_layer_prev_shots.shape[0],
                                                  dst_layer_cur_shots.shape[0])

                    self.__synapses_matrices[cur_popul_idx][cur_layer_idx][idx][0] = \
                        self.__update_synapses_matrix(shots_matrix, matrix)

                # Update prev_shots
                self.__prev_shots[cur_popul_idx][cur_layer_idx] = self.__current_shots[
                                                                      cur_popul_idx][
                                                                      cur_layer_idx] * (
                                                                          1 -
                                                                          syn_mem_fac) + \
                                                                  cur_layer_prev_shots \
                                                                  * syn_mem_fac


    def __update_synapses_matrix(self, shots_mat, cur_synapses_mat):
        # Update only on some probability. Otherwise leave the synapse unchanged by
        # setting the updation for it to zero
        ones_idxs = (shots_mat == 1)
        neg_ones_idxs = (shots_mat == -1)

        # Determine who will get increased by probability
        shots_mat[ones_idxs] *= (
                np.random.uniform(
                    size=(np.count_nonzero(ones_idxs))) < self.__change_prob_increase)
        # Determine who will get decreased by probability
        shots_mat[neg_ones_idxs] *= (
                np.random.uniform(
                    size=(np.count_nonzero(neg_ones_idxs))) < self.__change_prob_decrease)

        weighted_synapses_matrix = np.multiply(shots_mat, cur_synapses_mat)
        updated_mat = self.__create_update_mat(
            weighted_synapses_matrix) + cur_synapses_mat

        # Normalize the input synapses to each neuron
        columns_sum_cur_mat, column_sum_updated_mat = cur_synapses_mat.sum(
            axis=0), updated_mat.sum(axis=0)
        # If the column sum is 0 then we will divide by 1 and still remain with 0's
        return updated_mat / np.where((columns_sum_cur_mat != 0) & (column_sum_updated_mat
                                                                    != 0),
                                      (column_sum_updated_mat /
                                       columns_sum_cur_mat), 1)


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
        frame = self.__visualization_frame
        show_during_the_run = True
        h, w, _ = frame.shape

        # If recording:
        record = self.__vis_record[0]
        if record:
            show_during_the_run = self.__vis_record[1]
            out = self.__record_writer

        neuron_size = 5
        line_thick = 1

        # Decrease the neuron size to keep the avoid cutting of the neurons
        w, h = w - 2 * neuron_size, h - 2 * neuron_size

        # Each popul gets an equal part of the frame
        popul_gap = w // len(self.__layers)

        for popul_idx, population_neurons in enumerate(self.__layers):
            layers_num = len(population_neurons)
            layer_gap_y = h // layers_num
            layer_gap_x = popul_gap // layers_num

            for cur_layer_idx, cur_layer in enumerate(population_neurons):
                neurons_gap = layer_gap_y // (len(cur_layer) + 1)

                for neuron_idx, neuron in enumerate(cur_layer):
                    location = self.__calc_location(cur_layer_idx, layer_gap_x,
                                                    layer_gap_y, neuron_idx,
                                                    neuron_size,
                                                    neurons_gap, popul_gap, popul_idx)
                    shot_flag = self.__current_shots[popul_idx][cur_layer_idx][neuron_idx]

                    # First draw connections to all other neurons
                    self.__draw_connections_from_layer(cur_layer_idx, frame, h,
                                                       line_thick, location, neuron_idx,
                                                       neuron_size, popul_gap, popul_idx,
                                                       shot_flag)

                    # now draw the neuron
                    self.__draw_neuron(frame, location, neuron / self.__neurons_max_res,
                                       neuron_idx,
                                       neuron_size,
                                       popul_idx)

                    # In case of debugging, see each step
                    if debug:
                        cv2.imshow(self.__vis_window_name, frame)
                        cv2.waitKey(0)

        if show_during_the_run:
            cv2.imshow(self.__vis_window_name, frame)
            cv2.waitKey(1)

        if record:
            save_frame = (frame * 255).astype(np.uint8)
            out.write(save_frame)


    def __draw_neuron(self, frame, location, neuron_draw_val, neuron_idx, neuron_size,
                      popul_idx):
        color = (neuron_draw_val, neuron_draw_val, neuron_draw_val)
        cv2.circle(frame, location, neuron_size,
                   color,
                   thickness=-1)
        # Show the IINs by drawing a circle around them
        if neuron_idx >= self.__IINs_start_per_popul[popul_idx]:
            cv2.circle(frame, location, neuron_size, (0, 0, 1), thickness=1)


    def __calc_location(self, cur_layer_idx, layer_gap_x, layer_gap_y, neuron_idx,
                        neuron_size, neurons_gap, popul_gap, popul_idx):
        neuron_gap_x = layer_gap_x // 2
        layer_loc_x = popul_idx * popul_gap + cur_layer_idx * layer_gap_x

        # neuron_size - to draw all the neurons on the viewable frame
        # layer_loc_x - the x coming from the location of the neuron in the
        # population-layer hierarchy
        # neuron_gap_x * (neuron_idx % 2) - to show neurons in the same layer in a
        # levered way.
        return (neuron_size + layer_loc_x + neuron_gap_x * (neuron_idx % 2),
                neuron_size +
                cur_layer_idx *
                layer_gap_y +
                neuron_idx * neurons_gap)


    def __draw_connections_from_layer(self, cur_layer_idx, frame, h, line_thick, location,
                                      neuron_idx, neuron_size, popul_gap, popul_idx,
                                      shot_flag):
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
                to_location = self.__calc_location(to_layer_idx, to_layer_gap_x,
                                                   to_layer_gap_y, to_neuron_idx,
                                                   neuron_size,
                                                   to_neurons_gap, popul_gap,
                                                   to_popul_idx)
                # Draw the connections
                connections_strength = mat[neuron_idx, to_neuron_idx]
                if connections_strength:
                    # The color will be converted from [0,1] to [0,255], that's why the
                    # normalization with the "connections_max_res"

                    # If it shot notify it in the visualization
                    if shot_flag:
                        cv2.line(frame, location, to_location,
                                 (1, 0, 0), line_thick)
                    else:
                        cv2.line(frame, location, to_location,
                                 (
                                     0, connections_strength / self.__connections_max_res,
                                     0),
                                 line_thick)


if __name__ == '__main__':
    N = 4
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(1, 1), (1, 1), (1, 1)]
    configuration = {BrainNN.NODES_DETAILS: nodes_details,
                     BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                     BrainNN.VISUALIZATION_FUNC_STR: 'No ',
                     BrainNN.SYNAPSES_INITIALIZE_MEAN: 100,
                     BrainNN.SHOOT_THRESHOLD: 100}
    brainNNmodel = BrainNN(configuration)
    brainNNmodel.visualize()
    cv2.waitKey()
