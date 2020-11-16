# Writer: Gal Harari
# Date: 22/07/2020
from brainNN import BrainNN, NetWrapper
import numpy as np

N = 4


def get_binary_rep(val, noise_std=0):
    binary_represnt_list = [(val >> i) & 1 for i in range(N - 1, -1, -1)]
    binary_represnt_np = np.array(binary_represnt_list)
    return np.abs(np.random.normal(binary_represnt_np, noise_std))


def create_binary_input_generator(inject_answer=True, cycles=1):
    current_num = 1
    shots_count = 0
    input_amp = 15
    noise_std = 0.1 / input_amp
    cycles_counter = 0
    # the max part of the shooting threshold the injection is willing to inject
    inj_lim_from_thresh = 0.9

    identified_input_shots_needed = 5
    last_popul_inject = [np.zeros(2 ** N - 1), None]
    sensory_input = get_binary_rep(current_num, noise_std) * input_amp


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
            print(f"Shot on {current_num}")

        # If it identified the number, move on to the next one
        if shots_count >= identified_input_shots_needed:
            # Reset the shot count
            shots_count = 0

            # Move to the next number
            current_num = (current_num + 1) % (2 ** N)
            # Avoid inserting 0 when finishing a cycle
            if current_num == 0:
                print(f"Cycle number {cycles_counter}")
                current_num += 1
                cycles_counter += 1

            print(f"Current Input change to: {current_num}")

            # Create the new sensory input
            sensory_input = get_binary_rep(current_num, noise_std) * input_amp

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
        return cycles_counter < cycles


    # Update the sensory input to the first layer in the first population

    # Update the inject arrays to the last population
    return input_generator


def evaluate_binary_representation_nn(net, sequential=True, noise=0, inp_amp=15):
    net.zero_neurons()
    net_wrapper = NetWrapper(net, noise_std=noise/inp_amp)
    for i in range(1, (2 ** N)):
        x = get_binary_rep(i) * inp_amp
        y = i - 1
        output = net_wrapper(x)
        pred_y = np.argmax(output)
        print(f"Ground truth: {y}| Output: {pred_y} | Output vector: {output}")

        if not sequential:
            net.zero_neurons()


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
