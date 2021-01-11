# Writer: Gal Harari
# Date: 15/07/2020
import numpy as np
import cv2
import os.path
import dill
from src.utils.general_utils import save_json, load_json

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
                              'weaker exponentially by this factor. It also affects the' \
                              ' inner connections in the same way, and make the nodes ' \
                              'connections stronger by this factor. in (1,inf]'
    IINS_STRENGTH_FACTOR = 'The IINs will stronger by this factor than the excitatory'
    SHOOT_THRESHOLD = 'Threshold that above the neuron will shoot'
    FEEDBACK = 'Flag (True or False) to say if there will be connections from a ' \
               'population to lower populations'
    SPACIAL_ARGS = 'should be only changed in case of images. in that case insert ' \
                   '(<rows_num>,<columns_num>)'
    SYNAPSE_SPACIAL_DISTANCE_FACTOR = 'Determines the spacial strength insize inner ' \
                                      'layer connections. The bigger it is the weaker ' \
                                      'connections from far neurons will be. in (1,inf].'
    SYNAPSE_INCREASE_PROBABILITY = 'Probability for a synapse weight to INCREASE in ' \
                                   'case it provides the learning rule'
    SYNAPSE_DECREASE_PROBABILITY = 'Probability for a synapse weight to DECREASE in ' \
                                   'case it provides the learning rule'
    SYNAPSE_MEMORY_FACTOR = 'The synapses are updated according to shots into them in ' \
                            'previous steps. Thus the previous shots into a layer is a ' \
                            'moving average of the shots into this layer. Each time ' \
                            'step advancing, the previous average is multiplied by THIS' \
                            ' FACTOR and added to the current shots mulitplied by (' \
                            '1-THIS FACTOR). RANGE: [0,1]. higher than 0.5 will cause ' \
                            'it to remember previous shots, lower will forget prev shots'
    SYNAPSE_INCREASE_FUNC = 'Function that gets (np array of the weights) and returns a' \
                            ' np array of addon to add to the weights. Operate on ' \
                            'weights that increases after an iteration.'
    SYNAPSE_DECREASE_FUNC = 'Function that gets (np array of minus the weights) and ' \
                            'returns a np array of negative addon to add to the ' \
                            'weights. Operate on weights that decreases after an ' \
                            'iteration.'
    VISUALIZATION_FUNC_STR = 'Visualization function to use'
    VISUALIZATION_SIZE = "Visualization image's size"
    INTER_CONNECTIONS_PER_LAYER = 'Stats for each layer if it has inter-connections'
    RECORD_FLAG = 'Record the chosen visualization of the run. Also allows to disable ' \
                  'viewing if recording. [to_record_, to_show_during_run]'
    SAVE_SYNAPSES = 'Saved Synapses associated string'
    SAVE_NEURONS = 'Saved layers\' neurons associated string'

    # The parameters requires to save a net in the configuration arguments
    _required_to_save = [NODES_DETAILS, IINS_PER_LAYER_NUM]

    # Layers types:
    RF = 'Receptive-Field Synapses'
    FC = 'Fully Connected Synapses'

    default_configuration = {
        # Neuron parameters
        # [0][0] layer is the input layer. [-1][0] layer is the output layer
        NODES_DETAILS: [4, 10, 10],
        # This also sets the number of layers in population
        IINS_PER_LAYER_NUM: [(2, 1), (1, 3), (1, 1)],
        # In order for each population setup regarding the inter-connections to count,
        # it has to be the same number of layers in it.
        INTER_CONNECTIONS_PER_LAYER: [(False, True), (True, False), (False, False)],
        # This might be connected to " WEIGHTS_SUM_INTO_NEURON "
        SYNAPSES_INITIALIZE_MEAN: 5,
        SYNAPSES_INITIALIZE_STD: 0.15,
        # Should be lower than 1 if the synapses mean is lower than 1!
        SYNAPSE_DISTANCE_FACTOR: 3,
        IINS_STRENGTH_FACTOR: 2,
        SHOOT_THRESHOLD: 40,
        FEEDBACK: False,
        SPACIAL_ARGS: (1, 1),
        SYNAPSE_SPACIAL_DISTANCE_FACTOR: 3,
        SYNAPSE_INCREASE_PROBABILITY: 0.8,
        SYNAPSE_DECREASE_PROBABILITY: 0.7,
        SYNAPSE_MEMORY_FACTOR: 0.6,
        SYNAPSE_INCREASE_FUNC: lambda weights: np.minimum(weights / 2, 0.04),
        SYNAPSE_DECREASE_FUNC: lambda neg_weights: np.maximum(neg_weights / 2, -0.04),
        VISUALIZATION_FUNC_STR: 'None',
        # [0] is width, [1] is height
        VISUALIZATION_SIZE: [700, 1300],
        RECORD_FLAG: [False, False]

    }


    def __init__(self, conf_args={}):
        self._conf_args = {key: conf_args.get(key, BrainNN.default_configuration[key])
                           for key in BrainNN.default_configuration.keys()}

        self._thresh = self._conf_args[BrainNN.SHOOT_THRESHOLD]
        self._freeze = False

        self._feedback = self._conf_args[BrainNN.FEEDBACK]

        self._syn_memory_factor = self._conf_args[BrainNN.SYNAPSE_MEMORY_FACTOR]
        self._syn_inc_func = self._conf_args[BrainNN.SYNAPSE_INCREASE_FUNC]
        self._syn_inc_prob = self._conf_args[BrainNN.SYNAPSE_INCREASE_PROBABILITY]
        self._syn_dec_func = self._conf_args[BrainNN.SYNAPSE_DECREASE_FUNC]
        self._syn_dec_prob = self._conf_args[BrainNN.SYNAPSE_DECREASE_PROBABILITY]
        self._validate_learning_functions()

        # Visualize section
        # Create visualization frame
        self._visualize_dict = {
            'default': self._default_visualize,
            'light': self._light_visualize,
            'None': (lambda: None)
        }
        vis_size = self._conf_args[BrainNN.VISUALIZATION_SIZE]
        self._visualization_frame = np.zeros((vis_size[0], vis_size[1], 3))
        h, w, _ = self._visualization_frame.shape
        # Choose visualization function
        self.set_visualization(self._conf_args[BrainNN.VISUALIZATION_FUNC_STR])
        self._vis_counter = 0

        # Create visualization window
        self._vis_window_name = VISUALIZATION_WINDOW_NAME
        cv2.namedWindow(self._vis_window_name)
        # To make sure it's in the viewable screen
        cv2.moveWindow(self._vis_window_name, 0, 0)
        self._vis_record = self._conf_args[BrainNN.RECORD_FLAG]
        assert len(self._vis_record) == 2, "RECORD_FLAG format must be [to_record, " \
                                           "to_show_during_run]"
        self._record_writer = None
        if self._vis_record[0]:
            self._create_video_writer()

        # Initialize records structures
        self._sensory_input = None
        self._inject_to_last_popul = None
        self._symbolic_shots = None
        self._init_model()
        self._init_sensory_input()

        # Those should be the max expected value of the connections and of the neurons.
        # It's the max value from which every higher value will be with the same color
        # in the visualization
        self._connections_max_res = self._determine_weights_res()
        self._neurons_max_res = self._thresh


    def _create_video_writer(self):
        h, w, _ = self._visualization_frame.shape
        self._record_writer = cv2.VideoWriter(RECORD_SAVE_NAME + '.avi',
                                              cv2.VideoWriter_fourcc(*'XVID'),
                                              10, (w, h))


    def _determine_weights_res(self):
        max_value = 0.
        # Iterate only on synapses from the first layer, assuming it represents well
        # all the others
        for mat, idx in self._synapses_matrices[0][0]:
            mat_max = np.max(np.abs(mat))
            max_value = mat_max if mat_max > max_value else max_value

        # That's not accurate! It's not that a weights can only increase by factor of 2,
        # But that's the approximation
        return max_value * 2


    def _init_model(self):
        excitatory_details, IINs_per_layer = self._conf_args[BrainNN.NODES_DETAILS], \
                                             self._conf_args[
                                                 BrainNN.IINS_PER_LAYER_NUM]
        assert len(excitatory_details) == len(IINs_per_layer), "Population number " \
                                                               "doesn't match between " \
                                                               "the layers details and " \
                                                               "the IINs details"

        # each sub-list is a population
        self._neurons_per_layer = [[exitatories + IINs[i] for i in range(len(IINs))]
                                   for
                                   exitatories, IINs in list(zip(excitatory_details,
                                                                 IINs_per_layer))]

        self._IINs_start_per_popul = excitatory_details[:]

        # each sub-list is a population
        self._layers = BrainNN._create_layers_np_arrays(self._neurons_per_layer)

        # Required for iterating
        # This is what is added to the layers after each time step
        self._change_in_layers = BrainNN._create_layers_np_arrays(
            self._neurons_per_layer)
        self._prev_shots = BrainNN._create_layers_np_arrays(self._neurons_per_layer)
        self._current_shots = BrainNN._create_layers_np_arrays(self._neurons_per_layer)

        self._create_connections()


    @staticmethod
    def _create_layers_np_arrays(neurons_per_layer):
        return [[np.zeros(population_neurons[i]) for i in range(len(
            population_neurons))] for population_neurons in neurons_per_layer]


    def _create_connections(self):
        # Initialize the connections with some randomness
        mean = self._conf_args[BrainNN.SYNAPSES_INITIALIZE_MEAN]
        std = self._conf_args[BrainNN.SYNAPSES_INITIALIZE_STD] * mean
        IINs_factor = self._conf_args[BrainNN.IINS_STRENGTH_FACTOR]

        # This determines for each layer if it will have inter-connections
        inter_connections_flags = self._conf_args[
            BrainNN.INTER_CONNECTIONS_PER_LAYER]

        # Arranged in [ population list[ layer list[matrices from layer to other
        # ones:= [matrix, (popul_dst_idx, layer_dst_idx) ] ] ]
        self._synapses_matrices = []
        for popul_idx, population_neurons in enumerate(self._neurons_per_layer):
            population_list = []
            self._synapses_matrices.append(population_list)

            for cur_layer_idx, layer_neurons_num in enumerate(population_neurons):
                layer_list = []
                population_list.append(layer_list)

                # Now iterate over all layers and create connections. We will create
                # only forward connections

                # Normalize each neuron's output. It depends only on the weights in the
                # matrices from its layer to others, only on its relevant row on these
                # matrices. So normalize the values in each row over the matrices.
                normalize_vec = np.zeros((layer_neurons_num,))

                # Allow backwards connections
                first_popul_to_conn = popul_idx
                if self._feedback:
                    first_popul_to_conn = 0

                for popul_idx_to_connect in range(first_popul_to_conn,
                                                  len(self._neurons_per_layer)):
                    popul_to_connect = self._neurons_per_layer[popul_idx_to_connect]

                    # Determine inter-connections inside the population's layers

                    popul_layers_inter_conns = self._validate_inter_connections_format(
                        popul_idx, popul_to_connect, inter_connections_flags)

                    for layer_to_conn_idx, layer_to_conn in enumerate(popul_to_connect):
                        # Create connections to layer "layer_to_connect" from current
                        # layer
                        if self._create_connections_between_2_layers(layer_neurons_num,
                                                                     layer_to_conn,
                                                                     layer_list,
                                                                     cur_layer_idx,
                                                                     layer_to_conn_idx,
                                                                     popul_idx,
                                                                     popul_idx_to_connect,
                                                                     mean, std,
                                                                     popul_layers_inter_conns):
                            normalize_vec += np.abs(layer_list[-1][0]).sum(axis=1)
                # The IINs might be stronger than the excitatory
                normalize_vec[self._IINs_start_per_popul[popul_idx]:] /= IINs_factor

                # Now normalize the output from each neuron as explained above the loop
                for i in range(len(layer_list)):
                    mat = layer_list[i][0]
                    layer_list[i][0] = self._thresh * mat / normalize_vec[:, np.newaxis]


    def _create_connections_between_2_layers(self, layer_neurons_num,
                                             l_to_conn_neurons_n,
                                             layer_list,
                                             cur_layer_idx,
                                             layer_to_conn_idx, popul_idx,
                                             popul_idx_to_connect, mean, std,
                                             popul_layers_inter_conns):
        # Don't create connections to different layers in different
        # populations
        if layer_to_conn_idx != cur_layer_idx and popul_idx_to_connect != popul_idx:
            return False

        idxs = tuple([popul_idx_to_connect, layer_to_conn_idx])
        layer_list.append([np.random.normal(mean, std, (layer_neurons_num,
                                                        l_to_conn_neurons_n)),
                           idxs])

        IINs_start_idx = self._IINs_start_per_popul[popul_idx]

        # Allow inhibition
        layer_list[-1][0][IINs_start_idx:] *= -1

        # Prevent self loops
        if layer_to_conn_idx == cur_layer_idx and popul_idx_to_connect == popul_idx:
            # Here it means the layers are the same layer

            # Check if it is defined to have inter-connections
            # Default is to NOT have inter-connections, so if it's not
            # defined well it will not have inter-connections
            if popul_layers_inter_conns is None or not popul_layers_inter_conns[
                cur_layer_idx]:
                # Here it is set to not have inter-connections. The IINs are the only
                # neurons to connect to the others
                layer_list[-1][0][:IINs_start_idx, :IINs_start_idx] = 0
                return True

            # layer_list[-1][0] is the matrix. layer_list[-1][1] are the idxs
            # Prevent self loops.
            np.fill_diagonal(layer_list[-1][0], 0)
        else:
            # Here it means those are 2 different layers

            # In the same population create only between nodes
            if popul_idx_to_connect == popul_idx:
                inner_non_IINS_mat = layer_list[-1][0][:IINs_start_idx, :IINs_start_idx]
                idxs_to_delete_in_non_IIN_part = ~np.eye(inner_non_IINS_mat.shape[0],
                                                         dtype=bool)
                # Delete all but diagonal in the non IINs part of the matrix
                layer_list[-1][0][:IINs_start_idx, :IINs_start_idx][
                    idxs_to_delete_in_non_IIN_part] = 0

            # Prevent IINs of one layer to shoot to another
            layer_list[-1][0][IINs_start_idx:, :] = 0

        # Weaken links between far populations, and strength inner
        # population connections
        dist_fac = self._conf_args[BrainNN.SYNAPSE_DISTANCE_FACTOR]
        spacial_dist_fac = self._conf_args[BrainNN.SYNAPSE_SPACIAL_DISTANCE_FACTOR]

        # for the same layer treat differently:
        if popul_idx_to_connect == popul_idx:
            # connections in the same layer get weaken as the neurons are far from each
            # other, and node connections get stronger
            # It's the same LAYER and not POPULATION since all this loop does for
            # different layers in the same population is strengthen the nodes connecitons

            # only mess with the excitatory neurons
            extory_num = self._IINs_start_per_popul[popul_idx]

            # For case of images to insert spacial meaning in the first population,
            # else it is always 1,1
            rows, cols = 1, 1
            if popul_idx == 0:
                rows, cols = self._conf_args[BrainNN.SPACIAL_ARGS]

            size = layer_list[-1][0].shape
            mult_mat = np.ones(size)
            for i in range(size[0]):
                for j in range(size[1]):
                    if i < extory_num and j < extory_num:
                        mult_mat[i, j] *= spacial_dist_fac ** (
                                1 - distance(i, j, rows, cols))

            layer_list[-1][0] *= mult_mat
        else:
            # connections between layers
            layer_list[-1][0] *= (dist_fac ** (1 - abs(popul_idx_to_connect - popul_idx)))
        return True


    def create_RF_synapses(self, mean, std, k_size, stride, src_l_num, src_l_IIN_start,
                           dst_l_num, dst_l_IIN_start, on_centered=True):
        # First set the default value, it will shoot into dst IINs
        syn_mat = np.full((src_l_num, dst_l_num),
                          mean / 4, dtype=np.float64)
        syn_mat += np.random.normal(0, std, (src_l_num, dst_l_num))
        # Prevent src IINs to shoot into dst
        syn_mat[src_l_IIN_start:, :] = 0

        rows, cols = self._conf_args[BrainNN.SPACIAL_ARGS]

        # Assert the dimensions fit
        kernels_per_row = np.ceil(cols / stride)
        kernels_per_col = np.ceil(rows / stride)
        match_neurons_number = kernels_per_row * kernels_per_col
        error_str = ("Wrong dimensions for RF layer with %d required and %d in "
                     "parctice" % (match_neurons_number, dst_l_IIN_start))
        assert dst_l_IIN_start == match_neurons_number, error_str

        # The complication in the kernel creation is to create sharper kernel than the
        # trivial implementation np.linspace(mean, -mean / 2, num=k_size) + noise
        if on_centered:
            p_range = int(k_size * 2 / 3)
            n_range = k_size - p_range
            p_kernel = np.linspace(mean, mean / 4, num=p_range)
            n_kernel = np.linspace(-mean / 3, -mean * 3 / 4, num=n_range)
            kernel = np.hstack([p_kernel, n_kernel]) + np.random.normal(0, std, k_size)
        else:
            n_range = int(k_size * 2 / 3)
            p_range = k_size - n_range
            p_kernel = np.linspace(-mean, -mean / 4, num=n_range)
            n_kernel = np.linspace(mean / 3, mean * 3 / 4, num=p_range)
            kernel = np.hstack([p_kernel, n_kernel]) + np.random.normal(0, std, k_size)

        d_to_val = np.zeros((rows + cols,))
        d_to_val[:k_size] = kernel

        for i in range(dst_l_IIN_start):
            # Create receptive field for i neuron in the dst_layer
            for j in range(src_l_IIN_start):
                # Calculate location of the middle neuron
                calc_i = i
                y = calc_i // kernels_per_col
                calc_i = calc_i % kernels_per_col
                # Get x location
                x = calc_i

                middle_neuron = x * stride + y * stride * cols
                # i is the neuron in the src_layer, j is the corresponding neuron in
                # dst_layer, middle neuron is the middle of the corresponding kernel
                syn_mat[j, i] = d_to_val[int(distance(j, middle_neuron, rows, cols))]

        return syn_mat


    def _validate_inter_connections_format(self, popul_idx, popul_to_connect,
                                           popul_layers_inter_conns):
        # Make sure the dimensions are right
        if len(popul_layers_inter_conns) != len(self._layers) or \
                len(popul_layers_inter_conns[popul_idx]) != len(popul_to_connect):
            # It means the setup is not defined properly
            popul_layers_inter_conns = None
            print("The inter-connections flags are not the same dimension "
                  "as the layers in the model are!")
        else:
            # It means everything is fine
            popul_layers_inter_conns = popul_layers_inter_conns[popul_idx]
        return popul_layers_inter_conns


    def _validate_learning_functions(self):
        """
        Make sure the decrease function can't inverse a synapse sign, the increase
        functions is positive, and the decrease function preserve 0's
        :return:
        """
        check_vec = np.linspace(0.0,
                                2 * self._conf_args[BrainNN.SYNAPSES_INITIALIZE_MEAN],
                                num=100)

        # Assert the increase functions always increase
        positive = self._syn_inc_func(check_vec)
        assert np.all(positive >= 0), "Increase function is able to not increase"

        # Assert the decrease function is limited and won't inverse a synapse sign
        limited = self._syn_dec_func(check_vec * -1)
        assert np.all(
            limited + check_vec >= 0), "Decrease function is able to invert a synapse " \
                                       "role"

        # Assert the decrease function will turn 0 into 0 (it is the one that is
        # actiavted on 0)
        zero = self._syn_dec_func(np.array([0]))
        assert zero[0] == 0, "Decrease function doesn't turn 0 to 0"


    def _init_sensory_input(self):
        # The sensory input array. Only to the lower layer in the first population,
        # not to the IINs in it
        self._sensory_input = np.zeros(self._IINs_start_per_popul[0])

        # Arrays to inject to the last population. To all layers in the last population
        last_popul_excitatory_neurons_num = self._IINs_start_per_popul[-1]
        self._inject_to_last_popul = [
            np.zeros(last_popul_excitatory_neurons_num) for i
            in range(len(self._layers[-1]))]

        # Initialize no symbolic shots, None for each layer in the last popul
        self._symbolic_shots = [None for i in range(len(self._current_shots[-1]))]


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

        # Save self._layers, self._synapses_matrices
        save_dict = {
            BrainNN.SAVE_NEURONS: self._layers,
            BrainNN.SAVE_SYNAPSES: self._synapses_matrices
        }

        # Save everything defined by the class as required in saving and loading.
        # originally it was only NODES_DETAILS, IINS_PER_LAYER_NUM
        for argument in BrainNN._required_to_save:
            save_dict[argument] = self._conf_args[argument]

        save_json(save_dict, save_name)


    @staticmethod
    def load_model(conf_args=None, name=None):
        """
        load the model. Default name to load from is the default name to save with.
        :param conf_args: configuration arguments for building the new net. Only the
        neurons and synapses are loaded, the rest is being rebuild
        :param name: name to load from. Without suffix.
        :return:
        """
        load_name = name if name else SAVE_NAME
        loaded_dict = load_json(load_name)

        # Create the configuration dictionary for the network
        if conf_args is None:
            conf_args = {}

        # Update the fields defined by the class that must be saved and loaded
        for argument in BrainNN._required_to_save:
            conf_args[argument] = loaded_dict[argument]

        # Build the net from the dict
        net = BrainNN(conf_args)

        # Update the net's synapses and neurons
        # Load the synapses and make them into numpy matrices
        syn_mats_as_np = []
        syn_mats_as_list = loaded_dict[BrainNN.SAVE_SYNAPSES]
        for popul_syn_mats in syn_mats_as_list:
            syn_mats_as_np.append([])
            for layer_syn_mats in popul_syn_mats:
                syn_mats_as_np[-1].append([[np.array(mat), idxs] for mat, idxs in
                                           layer_syn_mats])

        net._synapses_matrices = syn_mats_as_np

        # Load the layers' neurons and make them into numpy array for every layer
        layers_as_list = loaded_dict[BrainNN.SAVE_NEURONS]
        net._layers = [[np.array(layer) for layer in popul] for popul in layers_as_list]

        # re-calculate the desired resolution
        net._connections_max_res = net._determine_weights_res()

        return net


    def train(self, input_generator):
        """
        Train the network using the input_generator.
        :param input_generator: a functions that gets the model, update it's
        sensory input and injection to the last population. It returns False in case of
        finish or failure
        :return:
        """
        while input_generator(self):
            self.step()

        self.save_state()


    def step(self):
        """
        Advance one time step of the network - inject the injection to last population if
        exists, insert the sensory input, advance the neurons and synapses one time
        step, update the weights and visualize the network.
        :return:
        """
        # Update the input and the injection to the last layer
        IINs_start_inp_popul = self._IINs_start_per_popul[0]
        self._layers[0][0][:IINs_start_inp_popul] += self._sensory_input

        IINs_start_out_popul = self._IINs_start_per_popul[-1]
        for i, layer in enumerate(self._layers[-1]):
            self._layers[-1][i][:IINs_start_out_popul] += self._inject_to_last_popul[i]

        # Iterate:
        self._iterate()

        # Change to symbolic shots vector, for weight updating process
        last_pop_IIN_start = self._IINs_start_per_popul[-1]
        tmp_cur_shots_last_popul = self._current_shots[-1].copy()
        for i in range(len(self._symbolic_shots)):
            if self._symbolic_shots[i] is not None:
                self._current_shots[-1][i][:last_pop_IIN_start] = self._symbolic_shots[i]

        # Update weights
        if not self._freeze:
            self._update_weights()

        # Change back to the real shots vector, after weight updating
        self._current_shots[-1] = tmp_cur_shots_last_popul

        self.visualize()


    def freeze(self):
        """
        Fix the synapses
        :return:
        """
        self._freeze = True


    def unfreeze(self):
        """
        Allow synapses plasticity
        :return:
        """
        self._freeze = False


    def zero_neurons(self):
        """
        Zero the neurons of the network, the current and previous shots, and the
        injection to the last population if exists
        :return:
        """
        # Zero the neurons
        for popul in self._layers:
            for layer in popul:
                layer.fill(0)

        # Zero the injection
        for i in range(len(self._inject_to_last_popul)):
            self._inject_to_last_popul[i].fill(0)

        # Zero the symbolic shots
        for i in range(len(self._symbolic_shots)):
            self._symbolic_shots[i] = None

        # Zero the current and previous shots
        for popul_idx, popul_shots in enumerate(self._current_shots):
            for i in range(len(popul_shots)):
                popul_shots[i].fill(0)
                self._prev_shots[popul_idx][i].fill(0)


    def set_sensory_input(self, arr):
        """
        Set the sensory input into the net
        :param arr: the new sensory input. Must be the same dimension as the first
        layer's excitatory part in the first population.
        :return:
        """
        assert arr.shape == self._sensory_input.shape, "Sensory input inserted is in " \
                                                       "wrong shape!"
        self._sensory_input = arr


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
        if len(arr_list) == len(self._inject_to_last_popul):
            for inject_arr_idx, inject_arr in enumerate(arr_list):
                # Allow not to inject to a specific layer
                if inject_arr is None:
                    continue
                if inject_arr.shape != self._inject_to_last_popul[inject_arr_idx].shape:
                    raise ValueError("inject to last popult during training with wrong "
                                     "dimension!")
                self._inject_to_last_popul[inject_arr_idx] = arr_list[inject_arr_idx]


    def set_symbolic_shots(self, symb_shots):
        # For None
        if symb_shots is None:
            return

        # Make sure all the dimension are right
        IINS_start = self._IINs_start_per_popul[-1]
        if len(symb_shots) == len(self._current_shots[-1]):
            self._symbolic_shots = symb_shots

            for symb_arr_idx, symb_arr in enumerate(symb_shots):
                # Allow not to inject to a specific layer
                if symb_arr is None:
                    continue
                if symb_arr.shape[0] != IINS_start:
                    raise ValueError(
                        "Symbolic shots during training with wrong dimension!")


    def get_shot_threshold(self):
        """
        :return: neurons voltage threshold it must pass to shoot.
        """
        return self._thresh


    def get_output(self, get_shots=False, whole_popul=False):
        """
        Return the output layers, only the excitatory neurons in it.
        :param whole_popul: if set to True, the whole last population is returned. In
        this case also the IINs in the last populations will be included
        :param get_shots: if set to True, the shots instead of the values will be returned
        :return: The output of the brainNN
        """
        last_popul = self._current_shots[-1] if get_shots else self._layers[-1]
        excit_neuron_n = self._IINs_start_per_popul[-1]
        if whole_popul:
            return [last_popul[i][:excit_neuron_n] for i in range(len(last_popul))]
        return last_popul[0][:excit_neuron_n]


    def get_sensory_input(self):
        """
        Only for debug purposes
        :return: the current sensory input to the net
        """
        return self._sensory_input


    def _iterate(self):
        """
        update all of the neurons and all of the connections one time step forward
        TODO: implement more efficient
        :return:
        """
        for cur_popul_idx, cur_popul in enumerate(self._layers):
            for cur_layer_idx, cur_layer in enumerate(cur_popul):
                self._current_shots[cur_popul_idx][cur_layer_idx] = (cur_layer >
                                                                     self._thresh) * 1
                cur_shots = self._current_shots[cur_popul_idx][cur_layer_idx]

                # Deduce from those who fired
                # Counting on that it resets after each time stamp
                self._change_in_layers[cur_popul_idx][
                    cur_layer_idx] -= cur_shots * cur_layer

                matrices_from_cur_layer = self._synapses_matrices[cur_popul_idx][
                    cur_layer_idx]

                for matrix, dst_idxs in matrices_from_cur_layer:
                    self._change_in_layers[dst_idxs[0]][
                        dst_idxs[1]] += cur_shots @ matrix

        # Update all layers and reset the "change_in_layers" help array
        for cur_popul_idx, cur_popul in enumerate(self._layers):
            for cur_layer_idx, cur_layer in enumerate(cur_popul):
                # Make sure the IINs doesn't diminish too much to negatate the value
                cur_layer += np.maximum(self._change_in_layers[cur_popul_idx][
                                            cur_layer_idx], -cur_layer)
                self._change_in_layers[cur_popul_idx][cur_layer_idx] *= 0


    def _update_weights(self):
        """
        TODO: implement more efficient. can be easily done by inserting it during the
              process of firing, as described in the doc.
        :return:
        """
        syn_mem_fac = self._syn_memory_factor
        # Iterate over the layers
        for cur_popul_idx, cur_popul in enumerate(self._layers):
            for cur_layer_idx in range(len(cur_popul)):
                # Get the shots of the current layer's neurons, in the previous time step
                cur_layer_prev_shots = self._prev_shots[cur_popul_idx][cur_layer_idx]

                # Get the matrices out of the current layer to other layers
                matrices_from_cur_layer = self._synapses_matrices[cur_popul_idx][
                    cur_layer_idx]

                for idx, mat_data in enumerate(matrices_from_cur_layer):
                    matrix, dst_idxs = mat_data
                    dst_layer_cur_shots = self._current_shots[dst_idxs[0]][dst_idxs[1]]

                    # Most of the times te weights won't change:
                    if dst_layer_cur_shots.any():
                        # who didn't previously shot get -1 to decrease if the dst neuron
                        # shot anyway. That's why ( 2 * cur_layer_prev_shots - 1 ) to
                        # make 1
                        # to 1 and 0 to -1. The cur_layer_prev_shots is a moving
                        # average of
                        # previous shots. Above 0.5 will count as it shot, bellow and
                        # equal
                        # will count as it didn't shot.
                        shots_matrix = np.outer(2 * (cur_layer_prev_shots > 0.5) - 1,
                                                dst_layer_cur_shots)
                        # I'm afraid it would be problematic so I add the assertion here
                        assert shots_matrix.shape == (cur_layer_prev_shots.shape[0],
                                                      dst_layer_cur_shots.shape[0])

                        self._synapses_matrices[cur_popul_idx][cur_layer_idx][idx][0] = \
                            self._update_synapses_matrix(shots_matrix, matrix)

                # Update prev_shots
                self._prev_shots[cur_popul_idx][cur_layer_idx] = self._current_shots[
                                                                     cur_popul_idx][
                                                                     cur_layer_idx] * (
                                                                         1 -
                                                                         syn_mem_fac) + \
                                                                 cur_layer_prev_shots \
                                                                 * syn_mem_fac


    def _update_synapses_matrix(self, shots_mat, cur_synapses_mat):
        # Update only on some probability. Otherwise leave the synapse unchanged by
        # setting the updation for it to zero
        ones_idxs = (shots_mat == 1)
        neg_ones_idxs = (shots_mat == -1)

        # Determine who will get increased by probability
        shots_mat[ones_idxs] *= (
                np.random.uniform(
                    size=(np.count_nonzero(ones_idxs))) < self._syn_inc_prob)
        # Determine who will get decreased by probability
        shots_mat[neg_ones_idxs] *= (
                np.random.uniform(
                    size=(np.count_nonzero(neg_ones_idxs))) < self._syn_dec_prob)

        # abs so We'll decrease only according to the learning rule, which is shots_mat
        weighted_synapses_matrix = np.multiply(shots_mat, np.abs(cur_synapses_mat))
        # Now decreasing from negative is adding positive
        addon_mat = self._create_update_mat(weighted_synapses_matrix) * np.sign(
            cur_synapses_mat)
        updated_mat = addon_mat + cur_synapses_mat

        # NOTICE! We assume here that the synapses change functions can't change the
        # weights sign, otherwise we would have zero what exceeded from the original value

        # Normalize the input synapses to each neuron
        columns_sum_cur_mat, column_sum_updated_mat = np.abs(cur_synapses_mat).sum(
            axis=0), np.abs(updated_mat).sum(axis=0)
        # If the column sum is 0 then we will divide by 1 and still remain with 0's
        return updated_mat / np.where((columns_sum_cur_mat != 0) & (column_sum_updated_mat
                                                                    != 0),
                                      (column_sum_updated_mat /
                                       columns_sum_cur_mat), 1)


    def _create_update_mat(self, weighted_syn_mat):
        return np.where(weighted_syn_mat > 0,
                        self._syn_inc_func(weighted_syn_mat),
                        self._syn_dec_func(weighted_syn_mat))


    def set_increase_func(self, increase_func):
        self._syn_inc_func = increase_func


    def set_increase_prob(self, increase_prob):
        self._syn_inc_prob = increase_prob


    def set_decrease_func(self, decrease_func):
        self._syn_dec_func = decrease_func


    def set_decrease_prob(self, decrease_prob):
        self._syn_dec_prob = decrease_prob


    def set_visualization(self, vis_func_str):
        """
        Sets the visualization function.
        :param vis_func_str: ID string of the desired visualization function,
        in the inner visualization functions dictionary.
        :return:
        """
        self.visualize = self._visualize_dict.get(vis_func_str,
                                                  self._default_visualize)


    def _default_visualize(self):
        """
        visualize the model according to it's current state and the visualization
        parameters chosen in the beginning.
        :return:
        """
        frame = self._visualization_frame
        show_during_the_run = True
        h, w, _ = frame.shape

        # If recording:
        record = self._vis_record[0]
        if record:
            show_during_the_run = self._vis_record[1]
            out = self._record_writer

        neuron_size = 5
        line_thick = 1

        # Decrease the neuron size to keep the avoid cutting of the neurons
        w, h = w - 2 * neuron_size, h - 2 * neuron_size

        self._draw_net(frame, h, line_thick, neuron_size, w)

        if show_during_the_run:
            cv2.imshow(self._vis_window_name, frame)
            cv2.waitKey(1)

        if record:
            save_frame = (frame * 255).astype(np.uint8)
            out.write(save_frame)


    def _draw_net(self, frame, h, line_thick, neuron_size, w, view_shots=True):
        # Each popul gets an equal part of the frame
        popul_gap = w // len(self._layers)
        for popul_idx, population_neurons in enumerate(self._layers):
            layers_num = len(population_neurons)
            layer_gap_y = h // layers_num
            layer_gap_x = popul_gap // layers_num

            for cur_layer_idx, cur_layer in enumerate(population_neurons):
                neurons_gap = layer_gap_y // (len(cur_layer) + 1)

                for neuron_idx, neuron in enumerate(cur_layer):
                    location = self._calc_location(cur_layer_idx, layer_gap_x,
                                                   layer_gap_y, neuron_idx,
                                                   neuron_size,
                                                   neurons_gap, popul_gap, popul_idx)
                    shot_flag = self._current_shots[popul_idx][cur_layer_idx][neuron_idx]
                    if not view_shots:
                        shot_flag = False

                    # First draw connections to all other neurons
                    self._draw_connections_from_layer(cur_layer_idx, frame, h,
                                                      line_thick, location, neuron_idx,
                                                      neuron_size, popul_gap, popul_idx,
                                                      shot_flag)

                    # now draw the neuron
                    self._draw_neuron(frame, location, neuron / self._neurons_max_res,
                                      neuron_idx,
                                      neuron_size,
                                      popul_idx)


    def _draw_neuron(self, frame, location, neuron_draw_val, neuron_idx, neuron_size,
                     popul_idx, shot_flag=False):
        if shot_flag:
            color = (1, 0, 0)
        else:
            color = (neuron_draw_val, neuron_draw_val, neuron_draw_val)
        cv2.circle(frame, location, neuron_size, color, thickness=-1)

        # Show the IINs by drawing a circle around them
        if neuron_idx >= self._IINs_start_per_popul[popul_idx]:
            cv2.circle(frame, location, neuron_size, (0, 0, 1), thickness=1)


    def _calc_location(self, cur_layer_idx, layer_gap_x, layer_gap_y, neuron_idx,
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


    def _draw_connections_from_layer(self, cur_layer_idx, frame, h, line_thick, location,
                                     neuron_idx, neuron_size, popul_gap, popul_idx,
                                     shot_flag):
        synapse_matrices = self._synapses_matrices[popul_idx][cur_layer_idx]
        for mat, idxs in synapse_matrices:
            to_popul_idx, to_layer_idx = idxs
            to_population_neurons = self._layers[to_popul_idx]

            layers_num = len(to_population_neurons)
            to_layer_gap_y = h // layers_num
            to_layer_gap_x = popul_gap // layers_num
            to_layer = to_population_neurons[to_layer_idx]
            to_neurons_gap = to_layer_gap_y // (len(to_layer) + 1)

            for to_neuron_idx, to_neuron in enumerate(to_layer):
                to_location = self._calc_location(to_layer_idx, to_layer_gap_x,
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
                    elif connections_strength < 0:
                        cv2.line(frame, location, to_location,
                                 (0,
                                  0,
                                  np.abs(connections_strength) /
                                  self._connections_max_res),
                                 line_thick)
                    else:
                        cv2.line(frame, location, to_location,
                                 (0,
                                  connections_strength / self._connections_max_res,
                                  0),
                                 line_thick)


    def _light_visualize(self):
        """
        visualize the model, updates every shot and every few steps
        :return:
        """
        frame = self._visualization_frame
        h, w, _ = frame.shape
        counter = self._vis_counter
        self._vis_counter = (self._vis_counter + 1) % 1000

        neuron_size = 5
        line_thick = 1

        # Decrease the neuron size to keep the avoid cutting of the neurons
        w, h = w - 2 * neuron_size, h - 2 * neuron_size

        if counter == 0:
            self._draw_net(frame, h, line_thick, neuron_size, w, view_shots=False)
        else:
            self._draw_shots(_, frame, h, neuron_size, w)

        cv2.imshow(self._vis_window_name, frame)
        cv2.waitKey(1)


    def _draw_shots(self, _, frame, h, neuron_size, w):
        popul_gap = w // len(self._layers)
        for popul_idx, population_neurons in enumerate(self._layers):
            layers_num = len(population_neurons)
            layer_gap_y = h // layers_num
            layer_gap_x = popul_gap // layers_num

            for cur_layer_idx, cur_layer in enumerate(population_neurons):
                if not (self._current_shots[popul_idx][cur_layer_idx].any() or
                        self._prev_shots[popul_idx][cur_layer_idx].any()):
                    continue

                neurons_gap = layer_gap_y // (len(cur_layer) + 1)

                for neuron_idx, neuron in enumerate(cur_layer):
                    shot_flag = self._current_shots[popul_idx][cur_layer_idx][
                        neuron_idx]
                    # If this neuron doesn't shoot and it didn't shot last round (and it
                    # needs to be reset in the frame) don't draw it
                    if not shot_flag:
                        if not self._prev_shots[popul_idx][cur_layer_idx][neuron_idx]:
                            continue

                    location = self._calc_location(cur_layer_idx, layer_gap_x,
                                                   layer_gap_y, neuron_idx,
                                                   neuron_size,
                                                   neurons_gap, popul_gap, popul_idx)

                    # now draw the neuron
                    self._draw_neuron(frame, location, 0,
                                      neuron_idx,
                                      neuron_size,
                                      popul_idx, shot_flag)


def distance(i, j, row_n, col_n):
    """
    calculate the distance between the location i and the location j in the matrix,
    considering it came from an image with row_n rows and col_n columns
    :param i:
    :param j:
    :param row_n:
    :param col_n:
    :return:
    """
    chn_size = row_n * col_n
    # Get z location
    z_i, z_j = i // chn_size, j // chn_size
    i, j = i % chn_size, j % chn_size

    # Get y location
    y_i, y_j = i // col_n, j // col_n
    i, j = i % col_n, j % col_n

    # Get x location
    x_i, x_j = i, j

    return np.linalg.norm([z_j - z_i, y_j - y_i, x_j - x_i], ord=1)


if __name__ == '__main__':
    N = 3
    nodes_details = [N, int(N / 4), int(N / 4), int(N / 5)]
    IINs_details = [(3, 3), (3, 3, 4), (3, 3), (1, 1)]
    inter_connections = [(True, True), (True, True), (True, True), (True, True)]
    spacial_args = (20, 20)
    feedback = True
    configuration = {BrainNN.NODES_DETAILS: nodes_details,
                     BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                     BrainNN.INTER_CONNECTIONS_PER_LAYER: inter_connections,
                     BrainNN.VISUALIZATION_FUNC_STR: 'No ',
                     BrainNN.SYNAPSES_INITIALIZE_MEAN: 100,
                     BrainNN.SHOOT_THRESHOLD: 100,
                     BrainNN.SPACIAL_ARGS: spacial_args,
                     BrainNN.FEEDBACK: feedback}
    brainNNmodel = BrainNN(configuration)
    brainNNmodel.visualize()
    cv2.waitKey()
    brainNNmodel.save_state()
    cv2.destroyAllWindows()

    loaded = BrainNN.load_model(configuration)
    loaded.visualize()
    cv2.waitKey()
