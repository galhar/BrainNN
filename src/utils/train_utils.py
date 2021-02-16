# Writer: Gal Harari
# Date: 18/11/2020
import numpy as np
import cv2
import random
from tqdm import tqdm


def np_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class NetWrapperBase:

    def __init__(self, net, req_shots_num, noise_std=0, verbose=False):
        self.brainNN = net
        self._n_std = noise_std
        self._req_shot_num = req_shots_num
        self._verbose = verbose


class TrainNetWrapper(NetWrapperBase):

    def __init__(self, net, req_shots_num, optimizer, noise_std=0,
                 verbose=False):
        super().__init__(net, req_shots_num, noise_std, verbose)
        self._optimizer = optimizer


    def __call__(self, inp, label):
        """
        Iterate over an input and a correct label. Calls it success when counted
        _req_shots_num times a correct shot. A 'correct shot' is when summing over the
        shots from last 'correct shot', and the argmax of this sum is the correct
        label. After each 'correct shot' it zeros the sum.
        :param inp:
        :param label:
        :return:
        """
        net = self.brainNN
        out_vec = np.zeros(net.get_output(get_shots=True).shape)
        correct_count = 0
        while correct_count < self._req_shot_num:
            # insert sensory input
            sensory_input = np.abs(np.random.normal(inp, self._n_std))
            net.set_sensory_input(sensory_input)
            self._optimizer.inject_label(label)

            net.step()

            # Get the output
            out_vec += net.get_output(get_shots=True)
            if np.argmax(out_vec) == label and out_vec.any():
                if self._verbose:
                    print(f"Shot on {label}")
                correct_count += 1
                # Zero it to restart the counting of a correct shot
                out_vec = np.zeros(net.get_output(get_shots=True).shape)


class EvalNetWrapper(NetWrapperBase):

    def __init__(self, net, noise_std=0, req_shots_num=5, norm_func=np_softmax):
        super().__init__(net, req_shots_num=req_shots_num, noise_std=noise_std)
        self._norm_func = norm_func


    def __call__(self, inp):
        """
        Finish iteration when there are more than req_shots_num from some label neuron
        :param inp:
        :return:
        """
        net = self.brainNN
        out_vec = np.zeros(net.get_output(get_shots=True).shape)
        while np.max(out_vec) < self._req_shot_num:
            # insert sensory input
            sensory_input = np.abs(np.random.normal(inp, self._n_std))
            net.set_sensory_input(sensory_input)

            net.step()

            # Get the output
            out_vec += net.get_output(get_shots=True)

        # Return normalize vec
        return self._norm_func(out_vec)


class DataLoaderBase:

    def __next__(self):
        """
        Returns batches of one epoch. After finished batches raise stopIteration
        :return: [batch of samples, batch of labels]
        """
        pass


    def __iter__(self):
        return self


class ClassesDataLoader(DataLoaderBase):

    def __init__(self, data_array, batched=False, shuffle=False,
                 noise_std=0):
        """

        :param data_array: [(<label>,<sample>),...]
        :param batched:
        :param shuffle:
        :param noise_std:
        """
        self._batched = batched
        self._shuffle = shuffle
        self._n_std = noise_std

        self._stopped_iter = True

        self.classes = [l for l, sample in data_array]
        self.classes_neurons = [i for i in range(len(data_array))]
        self.neuron_to_class_dict = {self.classes_neurons[i]: self.classes[i] for i in
                                     range(len(data_array))}

        self.samples = [self._noise(sample) for l, sample in data_array]

        # Batched options:
        self._pool_counter = 0
        self._idxs_pool = self.classes_neurons.copy()


    def _noise(self, s):
        size = s.shape
        return s + np.random.normal(0, self._n_std, size=size)


    def __next__(self):
        if not self._batched:
            return self._sequence_next()
        else:
            return self._batched_next()


    def _sequence_next(self):
        # create a single batch and raise stop iteration. If last time didn't raised
        # stopIteration than this one should do it
        if not self._stopped_iter:
            self._stopped_iter = True
            raise StopIteration
        # This "next" doesn't raise stopIteration
        self._stopped_iter = False

        if self._shuffle:
            p = np.random.permutation(len(self.classes_neurons))
            return [[self.samples[i] for i in p], [self.classes_neurons[i] for i in p]]

        return [self.samples, self.classes_neurons]


    def _batched_next(self):
        if self._pool_counter == len(self.classes_neurons):
            self._pool_counter = 0
            self._idxs_pool = self.classes_neurons.copy()
            raise StopIteration

        idx = random.choice(self._idxs_pool)
        self._idxs_pool.remove(idx)
        self._pool_counter += 1

        samples_batch = [self.samples[idx]]
        labels = [self.classes_neurons[idx]]
        return [samples_batch, labels]


class OptimizerBase:

    def __init__(self, net, increase_func, decrease_func, increase_prob,
                 decrease_prob, sample_reps, epochs):
        self._net = net
        self.inc_func = increase_func
        self.inc_prob = increase_prob
        self.dec_func = decrease_func
        self.dec_prob = decrease_prob
        self.sample_reps = sample_reps
        self.epochs = epochs


    def update_net(self):
        self._net.set_increase_func(self.inc_func)
        self._net.set_increase_prob(self.inc_prob)
        self._net.set_decrease_func(self.dec_func)
        self._net.set_decrease_prob(self.dec_prob)


    def inject_label(self, label):
        pass


class DefaultOptimizer(OptimizerBase):

    def __init__(self, net, sample_reps, epochs, sharp=False, inc_prob=0.7,
                 dec_prob=0.2):
        inc_func = lambda weights: np.minimum(np.minimum(weights / 10,
                                                         np.exp(-weights)), 0.04)
        dec_func = lambda neg_weights: np.maximum(neg_weights / 10, -0.04)
        OptimizerBase.__init__(self, net, inc_func, dec_func, inc_prob, dec_prob,
                               sample_reps, epochs)

        # inj_arr size of last_popul. Although we only use the first layer
        last_popul = self._net.get_output(whole_popul=True)
        self._inj_arr = np.zeros((len(last_popul), len(last_popul[0])))

        self._sharp = sharp
        lim_fac = 0.9
        if sharp:
            lim_fac = 0.99
        self._inj_lim = lim_fac * self._net.get_shot_threshold()


    def inject_label(self, l_neuron):
        """
        :param l_neuron: the neuron of the correct labeling
        :return:
        """
        output = self._net.get_output()
        idxs_without_cur_num = (np.arange(len(self._inj_arr[0])) != l_neuron)

        dec_fac, inc_fac = 0.1, 0.1
        if self._sharp:
            dec_fac = 1

        # Inject to teach the network
        # decrease from the wrong neurons
        self._inj_arr[0][idxs_without_cur_num] = - output[idxs_without_cur_num] * dec_fac
        # increase to the right neuron, make sure (0<inj && (inj + current<_inj_lim))
        inj = self._inj_lim if self._sharp else output[l_neuron] * inc_fac
        inj = min(inj, self._inj_lim - output[l_neuron])
        if inj < 0:
            inj = 0
        self._inj_arr[0][l_neuron] = inj

        self._net.set_last_popul_injection(self._inj_arr)


class Trainer:

    def __init__(self, net, data_loader, optimizer: OptimizerBase,
                 verbose=False):
        """

        :param net:
        :param data_loader:
        :param optimizer: in charge of the injection to the network, of setting the net
        synapses change functions and probabilities, setting the epoches and sample reps
        """
        self.net = net
        self._data_loader = data_loader
        self.optimizer = optimizer
        self._net_wrapper = TrainNetWrapper(net,
                                            req_shots_num=self.optimizer.sample_reps,
                                            optimizer=optimizer)
        self._hooks = []
        self.storage = {}

        self._verbose = verbose

        self.optimizer.update_net()


    def register_hook(self, hook):
        self._hooks.append(hook)


    def _build_hooks(self):
        hooks_list = []
        for hook in self._hooks:
            hooks_list.append(hook(self))
        self._hooks = hooks_list


    def train(self):
        self._build_hooks()

        for ep in range(self.optimizer.epochs):
            for sample_batch, labels in self._data_loader:
                # To allow progress bar
                samples_idxs = range(len(sample_batch))
                if self._verbose and not self._data_loader._batched:
                    samples_idxs = tqdm(samples_idxs)

                for i in samples_idxs:
                    sample, l = sample_batch[i], labels[i]

                    if self._verbose:
                        # Counting on that the loader is of type ClassesDataLoader
                        print(self._data_loader.neuron_to_class_dict[l])

                    self._net_wrapper(sample, l)

                self.net.zero_neurons()
                for h in self._hooks:
                    h.after_batch()
            print(f"Finished epoch {ep}\\{self.optimizer.epochs - 1}")
            for h in self._hooks:
                h.after_epoch()
