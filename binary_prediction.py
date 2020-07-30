# Writer: Gal Harari
# Date: 22/07/2020
from brainNN import BrainNN
import numpy as np

N = 5


def get_noisy_binary_rep(val, noise_std):
    binary_represnt_list = [(val >> i) & 1 for i in range(N - 1, -1, -1)]
    binary_represnt_np = np.array(binary_represnt_list)
    return np.abs(np.random.normal(binary_represnt_np, noise_std))


def create_binary_input_generator():
    current_num = 1
    shots_count = 0
    noise_std = 0.1
    cycles = 0
    identified_input_shots_needed = 5
    last_popul_inject = [np.zeros(2 ** N - 1), None]
    sensory_input = get_noisy_binary_rep(current_num, noise_std)


    def input_generator(brainNN):
        nonlocal shots_count
        nonlocal current_num
        nonlocal identified_input_shots_needed
        nonlocal sensory_input
        nonlocal noise_std
        nonlocal cycles
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
            current_num = (current_num + 1) % 2 ** N
            # Avoid inserting 0 when finishing a cycle
            if current_num == 0:
                current_num += 1
                cycles += 1

            print(f"Current Input change to: {current_num}")

            # Create the new sensory input
            sensory_input = get_noisy_binary_rep(current_num, noise_std)

        indexes_without_cur_num = (
                np.arange(len(last_popul_inject[0])) != current_num_idx)

        output = brainNN.get_output()
        # Inject to teach the network
        last_popul_inject[0] = output
        last_popul_inject[0][indexes_without_cur_num] = output[
                                                            indexes_without_cur_num] / 1.1
        last_popul_inject[0][current_num_idx] = output[current_num_idx] * 1.01
        brainNN.set_last_popul_injection(last_popul_inject)

        # Insert the sensory input
        brainNN.set_sensory_input(sensory_input)
        # Return false in case of finish or error
        return cycles < 1


    # Update the sensory input to the first layer in the first population

    # Update the inject arrays to the last population
    return input_generator


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
