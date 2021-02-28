# Writer: Gal Harari
# Date: 04/06/2020
import numpy as np
from tqdm import tqdm
import timeit
import cv2
import imutils
import cProfile
import pstats
import matplotlib.pyplot as plt
from src.brainNN import distance, BrainNN
from src.binary_encoding_task.main_binary import trainer_train
from src.Fonts_task.main_font import fonts_trainer_evaluation
from src.Fonts_task.font_prediction import MNISTDataLoader


def test_sum_np_vs_python():
    global x
    # x = list(range(1000))
    # x = range(1000)
    # or
    x = np.random.standard_normal(1000)


    def pure_sum():
        return sum(x)


    def numpy_sum():
        return np.sum(x)


    n = 10000
    t1 = timeit.timeit(pure_sum, number=n)
    print('Pure Python Sum:', t1)
    t2 = timeit.timeit(numpy_sum, number=n)
    print('Numpy Sum:', t2)


def test_threshold_1_vs_many_small():
    thresh = 0.5
    x_one = np.random.standard_normal(1000)
    x_many = np.random.standard_normal((5, 200))


    def thresh_one_big_array():
        return x_one > thresh


    def thresh_many_small_arrays():
        return x_many > thresh


    n = 1000000
    t1 = timeit.timeit(thresh_one_big_array, number=n)
    print('Threshing one big array takes:', t1)
    t2 = timeit.timeit(thresh_many_small_arrays, number=n)
    print('Threshing many small array takes', t2)


def test_reset_multiply_vs_indexing():
    thresh = 0.5
    x = np.random.standard_normal(1000)
    threshed = x > thresh


    # x = np.random.standard_normal((5, 200))
    # threshed = x > thresh

    def indexing():
        x[threshed] = 0
        return


    def multiply():
        b = x * threshed


    n = 1000000
    t1 = timeit.timeit(multiply, number=n)
    print('Reset to zero by multiplying:', t1)
    t2 = timeit.timeit(indexing, number=n)
    print('Reset to zero by indexing', t2)


frame = np.zeros((1000, 1000, 3))


def test_line_draw():
    n_lines = 200


    def draw_lines():
        global frame
        for i in range(n_lines):
            color = (255 - i, i, 0)

            # Line thickness of 9 px
            thickness = 1

            # Using cv2.line() method
            # Draw a diagonal green line with thickness of 9 px
            cv2.line(frame, (i, i), (1000 - i, 1000 - i), color, thickness)

            # Displaying the image


    # cv2.line(frame, (200, 200), (1000, 1000), (0,244,0), 3)
    t1 = timeit.timeit(draw_lines, number=1)
    print(f'Time to print {n_lines} lines is:', t1)

    cv2.imshow('frame', frame)
    cv2.waitKey()


def test_random():
    thresh = 0.5
    x = np.random.standard_normal((5, 200))
    threshed = x > thresh

    eta = 0.1

    N = 1000
    p = 0.3


    def random_synapses_update():
        updating_vec = np.random.choice(a=[False, True], size=(np.sum(threshed),),
                                        p=[p, 1 - p]) * eta
        # np.put(x, threshed, updating_vec)


    def random_using_binominal():
        updating_vec = np.random.binomial(1, p, size=(np.sum(threshed),)) * eta

        # np.put(x, threshed, updating_vec)


    def random_using_uniform_threshing():
        updating_vec = (np.random.uniform(size=(np.sum(threshed),)) < p) * eta

        # np.put(x, threshed, updating_vec)


    def check_adding_vector_instead_of_random():
        updating_vec = np.zeros((np.sum(threshed),))
        for i in range(int(1 / p)):
            updating_vec += 1
            updating_vec > 1


    n = 100000
    # for 100000 times it takes 3.7 to create random vec and 5.5 to update accordingly
    # thats not what I want but it does have the random here which takes A LOT of time
    t1 = timeit.timeit(random_synapses_update, number=n)
    print('Time to determine synapses updating using choice numpy:', t1)
    # it takes about half the previous
    t2 = timeit.timeit(random_using_binominal, number=n)
    print('Time to determine synapses updating using binom numpy:', t2)
    t3 = timeit.timeit(random_using_uniform_threshing, number=n)
    print('Time to determine synapses updating using uniform numpy:', t3)
    t4 = timeit.timeit(check_adding_vector_instead_of_random, number=n)
    print('Time to determine synapses updating using winners remembering:', t4)


def test_count_occurences():
    thresh = 0.5
    mat = np.random.standard_normal((5, 200))
    threshed = mat > thresh
    mat = 2 * mat - 1

    N = 1000


    def sum_with_equal():
        ones = (mat == 1).sum()
        negative_ones = (mat == -1).sum()


    def sum_with_greater_lower():
        ones = (mat > 0).sum()
        negative_ones = (mat < 0).sum()


    def count_zeros_with_equal():
        ones = np.count_nonzero(mat == 1)
        negative_ones = np.count_nonzero(mat == -1)


    def count_zeros_with_greater_lower():
        ones = np.count_nonzero(mat > 0)
        negative_ones = np.count_nonzero(mat < 0)


    n = 1000000
    # for 100000 times it takes
    t3 = timeit.timeit(count_zeros_with_equal, number=n)
    print('Time to count using count_zeros_with_equal:', t3)
    t4 = timeit.timeit(count_zeros_with_greater_lower, number=n)
    print('Time to count using count_zeros_with_greater_lower:', t4)
    t1 = timeit.timeit(sum_with_equal, number=n)
    print('Time to count using sum_with_equal:', t1)
    # it takes about half the previous
    t2 = timeit.timeit(sum_with_greater_lower, number=n)
    print('Time to count using sum_with_greater_lower:', t2)


def test_imshow_size():
    window_name = 'frame'
    frame = np.zeros((700, 1500, 3))
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)

    cv2.imshow(window_name, frame)
    cv2.waitKey()


def test_imshow_color():
    color = (0.4, 1, 0)
    made_frame = color * np.ones((1000, 1000, 3))
    img_frame = cv2.imread("test_image.jpg")
    img_frame = imutils.resize(img_frame, width=100)

    cv2.imshow('image frame', img_frame)
    cv2.imshow('made frame', made_frame)
    cv2.waitKey()


def test_write_video():
    frame = np.zeros((700, 1300, 3))
    h, w, _ = frame.shape
    out = cv2.VideoWriter('test_record_cv2' + '.avi',
                          cv2.VideoWriter_fourcc(*'XVID'),
                          10, (h, w))
    steps = 1000
    for i in range(steps):
        out.write(frame)
        frame[i:, :i, :] = i / steps
    out.release()


def vid_merge_example_with_test():
    frame = np.zeros((400, 700, 3))
    h, w, _ = frame.shape

    # Define the codec and create VideoWriter object.The output is stored in
    # 'outpy.avi' file.
    out = cv2.VideoWriter('test_record_cv2.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 100,
                          (w, h))

    steps = 400
    for i in range(steps):
        frame[i:, :i, 1:] = i / steps
        frame[i:, :i, :1] = 1 - i / steps

        # Write the frame into the file 'output.avi'
        save_frame = (frame * 255).astype(np.uint8)
        out.write(save_frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.waitKey(3)

        # When everything done, release the video capture and video write objects
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def vid_example():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system
    # dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in
    # 'outpy.avi' file.
    out = cv2.VideoWriter('video_saved_example.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame_width, frame_height))

    while (True):
        ret, frame = cap.read()

        if ret == True:
            # Write the frame into the file 'output.avi'
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

            # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def search_bottelneck():
    profile = cProfile.Profile()
    profile.runcall(fonts_trainer_evaluation)
    ps = pstats.Stats(profile)
    ps.sort_stats('time')
    ps.print_stats()


def check_distance(neuron_i, dist_fac=3, size=(3, 3), into_neuron=False):
    img = np.zeros(size)
    rows, cols = size
    img_flat = img.flatten()
    n = len(img_flat)
    syn_mat = np.ones((n, n))
    for i in tqdm(range(n)):
        for j in range(n):
            syn_mat[i, j] *= dist_fac ** (1 - distance(i, j, rows, cols))

    if neuron_i:
        if into_neuron:
            neurons_syn_vec = syn_mat[:, neuron_i]

        else:
            neurons_syn_vec = syn_mat[neuron_i, :]

        neuron_syns = neurons_syn_vec.reshape(size)

        plt.imshow(neuron_syns)
        plt.colorbar()
        plt.title(f"dist_fac = {dist_fac}")
        plt.show()
    else:
        for i in range(n):
            if into_neuron:
                neurons_syn_vec = syn_mat[:, i]

            else:
                neurons_syn_vec = syn_mat[i, :]

            neuron_syns = neurons_syn_vec.reshape(size)

            cv2.imshow('hi', neuron_syns)
            cv2.waitKey(10)


def check_RF_layers(k_size, stride, into_n=1, dst_n=400, rows=22, cols=22, show=True):
    mean, std = 3, 0.3
    iins = 12
    img_n = rows * cols
    src_l_num, src_l_IIN_start = img_n + iins, img_n
    dst_l_num, dst_l_IIN_start = dst_n + iins, dst_n
    on_centered = True
    white = True

    # <copyied from the BrainNN.py>:
    # First set the default value, it will fill the IINs
    syn_mat = np.full((src_l_num, dst_l_num),
                      mean / 4, dtype=np.float64)
    syn_mat += np.random.normal(0, std, (src_l_num, dst_l_num))
    # Prevent src IINs to shoot into dst
    syn_mat[src_l_IIN_start:, :] = 0
    cols = int(src_l_IIN_start / rows)

    # Assert the dimensions fit
    kernels_per_row = np.ceil(cols / stride)
    kernels_per_col = np.ceil(rows / stride)
    match_neurons_number = kernels_per_row * kernels_per_col * into_n
    error_str = ("Wrong dimensions for RF layer with %d required and %d in "
                 "practice" % (match_neurons_number, dst_l_IIN_start))
    assert dst_l_IIN_start == match_neurons_number, error_str

    if on_centered:
        p_range = int(k_size * 2 / 3)
        n_range = k_size - p_range
        p_kernel = np.linspace(mean, mean / 3, num=p_range)
        n_kernel = np.linspace(-mean / 3, -mean * 3 / 4, num=n_range)
        kernel = np.hstack([p_kernel, n_kernel]) + np.random.normal(0, std, k_size)

    else:
        n_range = int(k_size * 2 / 3)
        p_range = k_size - n_range
        p_kernel = np.linspace(-mean, -mean / 4, num=n_range)
        n_kernel = np.linspace(mean / 3, mean * 3 / 4, num=p_range)
        kernel = np.hstack([p_kernel, n_kernel]) + np.random.normal(0, std, k_size)
    if white:
        kernel = np.random.normal(mean, 0, k_size)
    d_to_val = np.zeros((rows + cols,))
    d_to_val[:k_size] = kernel

    for i in range(dst_l_IIN_start):
        # Create receptive field for i neuron in the dst_layer
        for j in range(src_l_IIN_start):
            # Calculate location of the middle neuron
            # Every <into_n> dst_neurons are counted as 1
            calc_i = i // into_n
            y = calc_i // kernels_per_col
            calc_i = calc_i % kernels_per_col
            # Get x location
            x = calc_i

            middle_neuron = x * stride + y * stride * cols
            # i is the neuron in the src_layer, j is the corresponding neuron in
            # dst_layer, middle neuron is the middle of the corresponding kernel
            syn_mat[j, i] = d_to_val[int(distance(j, middle_neuron, rows, cols))]
    kernel_idxs = (syn_mat != 0)
    syn_mat[kernel_idxs] += np.random.normal(0, std, syn_mat.shape)[kernel_idxs]
    # </End of copying>
    if show:
        for i in range(dst_l_IIN_start):
            if i < dst_l_num:
                # continue
                pass
            neurons_syn_vec = syn_mat[:, i]

            neuron_syns = neurons_syn_vec[:src_l_IIN_start].reshape((rows, cols))

            plt.imshow(neuron_syns)
            plt.colorbar()
            plt.pause(0.1)
            plt.show()
        plt.imshow(syn_mat)
        plt.colorbar()
        plt.show()
    return syn_mat[:src_l_IIN_start, :dst_l_IIN_start]


def compare_models_synapses():
    """
    made for debugging when I couldn't find the synapse difference after changing
    something in BrainNN the shouldn't have changed the synapses initialization.
    :return:
    """
    dir = "binary_encoding_task/"
    name1 = "prev_version_save.json"
    name2 = "experiment_save.json"
    syns1 = BrainNN.load_model(name=dir + name1)._synapses_matrices
    syns2 = BrainNN.load_model(name=dir + name2)._synapses_matrices

    for pop_i in range(len(syns1)):
        for l_i in range(len(syns1[pop_i])):
            for s_i in range(len(syns1[pop_i][l_i])):
                m1, idx1 = syns1[pop_i][l_i][s_i]
                m2, idx2 = syns2[pop_i][l_i][s_i]
                fig = plt.figure()
                fig.suptitle("")

                ax1 = fig.add_subplot(121)
                plt.imshow(m1)
                ax1.title.set_text(f"1 - {idx1}")

                ax1 = fig.add_subplot(122)
                plt.imshow(m2)
                ax1.title.set_text(f"2 - {idx2}")
                plt.show()
                input()


def zero_np(fac=0.99, initial=2, high_res=False):
    arr = np.zeros((4,))
    arr += initial
    i = 0
    if high_res:
        for i in range(100):
            print("i={}; Array is {}".format(i, arr))
            arr *= fac
    else:
        for i in range(20000):
            if i % 1000 == 999:
                print("Reached i={}; Array is {}".format(i, arr))
            arr *= fac


def test_shots_mat_by_syn_hist():
    syn_hist = np.array([[1, 2, 0.1, 4],
                         [3, 3, 4, 5],
                         [5, 4, 5, 6],
                         [-2, -3, -2, -1],
                         [-0.1, -3, -1.5, -1.2]])

    dst_layer_cur_shots = np.array([1, 0, 1, 0]).astype(np.bool)

    shots_mat = 2 * (syn_hist >=
                     (np.max(syn_hist, axis=0)[
                      :np.newaxis] * 2 / 3)) - 1
    neg_idxs = (syn_hist < 0)
    shots_mat[neg_idxs] = (2 * (syn_hist <= (np.min(
        syn_hist, axis=0)[:np.newaxis] * 2 / 3)) - 1)[neg_idxs]
    shots_mat[:, ~dst_layer_cur_shots] = 0
    print("Syn Hist:\n", syn_hist)
    print("Dst shots idxs:\n", dst_layer_cur_shots)
    print("Shots mat Generated:\n", shots_mat)


def visualize_images_layers(model_file=None):
    photos_n = 10
    data_loader = MNISTDataLoader(small=True)
    images, labels = data_loader.__next__()
    samples = list(zip(images, labels))
    img, label = samples[0]
    pop_num_to_check = 0
    output_img = get_layer_effect(img, model_file, pop_num_to_check)

    # PLOT
    fig = plt.figure()
    # Plot original
    for i, s in enumerate(samples):
        ax = fig.add_subplot(3, photos_n, i + 1)
        flat_img = s[0]
        img_size = int(np.sqrt(flat_img.shape[0]))
        plt.imshow(flat_img.reshape((img_size, img_size)))
        ax.title.set_text(f"{s[1]}")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Plot RF layer
    for i, s in enumerate(samples):
        ax = fig.add_subplot(3, photos_n, i + 1 + photos_n)
        output_img = get_layer_effect(s[0], model_file, 0)
        plt.imshow(output_img)
        ax.title.set_text(f"RF {s[1]}")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Plot output
    for i, s in enumerate(samples):
        ax = fig.add_subplot(3, photos_n, i + 1 + 2 * photos_n)
        output_img = get_layer_effect(s[0], model_file, 1)
        plt.bar([i for i in range(photos_n)], output_img[:photos_n])
        ax.title.set_text(f"{s[1]}")
        ax.set_yticklabels([])

    plt.show()


def get_layer_effect(img, model_file, pop_num_to_check):
    """
    if the layer to check will output something that isn't an image, it will output a
    1D vec which is it, without change.
    :param img:
    :param model_file:
    :param pop_num_to_check:
    :return:
    """
    if model_file is None:
        layer_mat_no_IINs = check_RF_layers(3, 1, 1, dst_n=144, rows=12, cols=12,
                                            show=False)
    else:
        model = BrainNN.load_model(name=model_file)
        mat = model._synapses_matrices[pop_num_to_check][0][1][0]
        src_IINs = model._IINs_start_per_popul[0]
        dst_IINs = model._IINs_start_per_popul[1]
        layer_mat_no_IINs = mat[:src_IINs, :dst_IINs]
    layer_output = img @ layer_mat_no_IINs
    img_size = int(np.sqrt(layer_output.shape[0]))
    if img_size * img_size == layer_output.shape[0]:
        output_img = layer_output.reshape((img_size, img_size))
    else:
        # Here it means it's just a layer without image meaning
        output_img = layer_output
    return output_img


if __name__ == '__main__':
    visualize_images_layers('Fonts_task/NetSavedByHookEp-7.json')
