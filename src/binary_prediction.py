# Writer: Gal Harari
# Date: 22/07/2020
from src.brainNN import BrainNN
from src.utils.train_utils import EvalNetWrapper
import numpy as np
from src.utils.train_utils import DataLoaderBase
import random
from deprecated import deprecated


N = 4


def get_binary_rep(val, amp, noise_std=0):
    binary_represnt_list = [(val >> i) & 1 for i in range(N - 1, -1, -1)]
    binary_represnt_np = np.array(binary_represnt_list)
    binary_represnt_np = binary_represnt_np / binary_represnt_np.sum() * amp
    return np.abs(np.random.normal(binary_represnt_np, noise_std))


class BinaryDataLoader(DataLoaderBase):

    def __init__(self, batched=False, shuffle=False, input_amplitude=30, noise_std=0):
        """
        :param input_amplitude:
        :param noise_std: This std is multiplied by the amplitude. Noise might cause the
        model to be more robust, like dropout in ANNs. Noise can be generated during
        training for each insertion of the input instead, using NetWrapper. Maybe it's
        better.
        """
        self.inp_amp = input_amplitude
        self.n_std = noise_std * input_amplitude
        self._stopped_iter = True
        self.classes = [i for i in range(2 ** N - 1)]
        self._cls_lst = self.classes.copy()
        self._cur_label = self.classes[0]
        self._batched = batched
        self._shuffle = shuffle


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
            random.shuffle(self._cls_lst)
        samples_batch = [get_binary_rep(i + 1, self.inp_amp, self.n_std) for
                         i in self._cls_lst]
        labels = self._cls_lst
        return [samples_batch, labels]


    def _batched_next(self):
        if self._cur_label == self.classes[-1] + 1:
            self._cur_label = self.classes[0]
            raise StopIteration

        samples_batch = [get_binary_rep(self._cur_label, self.inp_amp, self.n_std)]
        labels = [self._cur_label]
        self._cur_label += 1
        return [samples_batch, labels]


@deprecated(reason="This method isn't supported by the 'Trainer' hierarchy")
def create_binary_input_generator(inject_answer=True, repeat_sample=5, epoches=1,
                                  verbose=True, eval_hook=None):
    current_num = 1
    shots_count = 0
    input_amp = 15
    noise_std = 0 / input_amp
    cycles_counter = 0
    # the max part of the shooting threshold the injection is willing to inject
    inj_lim_from_thresh = 0.9

    identified_input_shots_needed = 1
    last_popul_inject = [np.zeros(2 ** N - 1), None]
    sensory_input = get_binary_rep(current_num, input_amp, noise_std)


    def input_generator(brainNN):
        nonlocal shots_count
        nonlocal current_num
        nonlocal identified_input_shots_needed
        nonlocal sensory_input
        nonlocal noise_std
        nonlocal cycles_counter
        # define the injection limit
        inj_lim = inj_lim_from_thresh * brainNN.get_shot_threshold()

        # Check for output shots
        current_num_idx = current_num - 1

        output_shots = brainNN.get_output(get_shots=True)
        # If the correct node shot, raise the shot count
        if output_shots[current_num_idx] == 1:
            shots_count += 1
            if verbose:
                print(f"Shot on {current_num}")

        # If it identified the number, move on to the next one
        if shots_count >= identified_input_shots_needed:
            # Reset the shot count
            shots_count = 0

            # Move to the next number
            current_num = (current_num + 1) % (2 ** N)
            # Avoid inserting 0 when finishing a cycle
            if current_num == 0:
                brainNN.zero_neurons()
                print(f"Finished Epoch {cycles_counter}")
                current_num += 1
                cycles_counter += 1

            if verbose:
                print(f"Current Input change to: {current_num}")

            # Create the new sensory input
            sensory_input = get_binary_rep(current_num, input_amp, noise_std)

        if inject_answer:
            indexes_without_cur_num = (
                    np.arange(len(last_popul_inject[0])) != current_num_idx)
            output = brainNN.get_output()

            # Inject to teach the network
            # decrease from the wrong neurons
            last_popul_inject[0][indexes_without_cur_num] = - output[
                indexes_without_cur_num] * 0.1
            # increase to the right neuron
            inj = output[current_num_idx] * 0.01
            last_popul_inject[0][current_num_idx] = inj if inj + output[
                current_num_idx] <= inj_lim else 0

            brainNN.set_last_popul_injection(last_popul_inject)

        # Insert the sensory input
        brainNN.set_sensory_input(sensory_input)
        # Return false in case of finish or error
        return cycles_counter < epoches


    # Update the sensory input to the first layer in the first population

    # Update the inject arrays to the last population
    return input_generator


def evaluate_binary_representation_nn(net, sequential=True, noise=0,
                                      inp_amp=15, req_shots=5):
    net.zero_neurons()
    # Number of shots it takes until the netWrapper will return the output vector of
    # the sum of all these shots normalized
    req_shots_for_decision = req_shots
    net_wrapper = EvalNetWrapper(net, noise_std=noise * inp_amp,
                                 req_shots_num=req_shots_for_decision)

    correct = 0
    total = 0
    for i in range(1, (2 ** N)):
        x = get_binary_rep(i, inp_amp)
        y = i - 1
        output = net_wrapper(x)
        pred_y = np.argmax(output)

        total += 1
        correct += 1 if pred_y == y else 0
        print(f"Ground truth: {y}| Output: {pred_y} | Output vector: {output}")
        if not sequential:
            net.zero_neurons()

    accuracy = 100 * correct / total
    print('Accuracy: %d %%' % (accuracy))
    return accuracy


if __name__ == '__main__':
    nodes_details = [N, 2 ** N, 2 ** N - 1]
    IINs_details = [(1, 1), (1, 1), (1, 1)]
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details}

    brainNN = BrainNN(configuration_args)
    input_generator = create_binary_input_generator()
    for i in range(10):
        input_generator(brainNN)
        print("BrainNN output: ", brainNN.get_output())
        print("BrainNN input: ", brainNN.get_sensory_input())
