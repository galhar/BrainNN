# Writer: Gal Harari
# Date: 18/11/2020
from brainNN import BrainNN
import numpy as np


def np_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class NetWrapperBase:

    def __init__(self, net: BrainNN, req_shots_num, noise_std=0, verbose=False):
        self.brainNN = net
        self._n_std = noise_std
        self._req_shot_num = req_shots_num
        self._verbose = verbose


class TrainNetWrapper(NetWrapperBase):

    def __init__(self, net: BrainNN, req_shots_num, optimizer, noise_std=0,
                 verbose=False):
        super().__init__(net, req_shots_num, noise_std, verbose)
        self._optimizer = optimizer


    def __call__(self, inp, label):
        net = self.brainNN
        out_vec = np.zeros(net.get_output(get_shots=True).shape)
        correct_count = 0
        while correct_count < self._req_shot_num:
            # insert sensory input
            sensory_input = np.abs(np.random.normal(inp, self._n_std))
            net.set_sensory_input(sensory_input)

            net.step()

            # Get the output
            out_vec += net.get_output(get_shots=True)
            if np.argmax(out_vec) == label:
                if self._verbose:
                    print(f"Shot on {label}")
                correct_count += 1

            self._optimizer.inject_label(label)


class EvalNetWrapper(NetWrapperBase):

    def __init__(self, net: BrainNN, noise_std=0,
                 req_shots_num=5, norm_func=np_softmax):
        super().__init__(net, req_shots_num=req_shots_num, noise_std=noise_std)
        self._norm_func = norm_func


    def __call__(self, inp):
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


class OptimizerBase:

    def __init__(self, net: BrainNN, increase_func, decrease_func, increase_prob,
                 decrease_prob, sample_reps, epoches):
        self._net = net
        self.inc_func = increase_func
        self.inc_prob = increase_prob
        self.dec_func = decrease_func
        self.dec_prob = decrease_prob
        self.sample_reps = sample_reps
        self.epoches = epoches


    def update_net(self):
        self._net.set_increase_func(self.inc_func)
        self._net.set_increase_prob(self.inc_prob)
        self._net.set_decrease_func(self.dec_func)
        self._net.set_decrease_prob(self.dec_prob)


    def inject_label(self, label):
        pass


class DefaultOptimizer(OptimizerBase):

    def __init__(self, net: BrainNN, sample_reps, epoches, inc_prob=0.7, dec_prob=0.2):
        OptimizerBase.__init__(self, net, lambda weights: np.full(weights.shape, 0.1),
                               lambda neg_weights: np.maximum(neg_weights / 2, -0.04),
                               inc_prob, dec_prob, sample_reps, epoches)
        # Injection array the same size as the last population. Most of the time we
        # only inject to the first layer of the last population
        last_popul = self._net.get_output(whole_popul=True)
        self._inj_arr = np.zeros((len(last_popul), len(last_popul[0])))
        self._inj_lim = 0.9 * self._net.get_shot_threshold()


    def inject_label(self, label_neuron):
        """
        :param label_neuron: the neuron of the correct labeling
        :return:
        """
        output = self._net.get_output()
        indexes_without_cur_num = (
                np.arange(len(self._inj_arr[0])) != label_neuron)

        # Inject to teach the network
        # decrease from the wrong neurons
        self._inj_arr[0][indexes_without_cur_num] = - output[
            indexes_without_cur_num] * 0.1
        # increase to the right neuron
        inj = output[label_neuron] * 0.01
        self._inj_arr[0][label_neuron] = inj if inj + output[
            label_neuron] <= self._inj_lim else 0

        self._net.set_last_popul_injection(self._inj_arr)


class Trainer:

    def __init__(self, net: BrainNN, data_loader, optimizer: OptimizerBase,
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
        self._net_wrapper = TrainNetWrapper(net, verbose=verbose,
                                            req_shots_num=self.optimizer.sample_reps,
                                            optimizer=optimizer)
        self._hooks = []
        self.storage = {}

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

        for ep in range(self.optimizer.epoches):
            for sample_batch, labels in self._data_loader:
                for i in range(len(sample_batch)):
                    sample, l = sample_batch[i], labels[i]
                    self._net_wrapper(sample, l)

                self.net.zero_neurons()
                for h in self._hooks:
                    h.after_batch()
            print(f"Finished epoch {ep}")
            for h in self._hooks:
                h.after_epoch()
