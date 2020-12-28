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
from src.brainNN import distance
from src.binary_encoding_task.main_binary import trainer_train


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
    profile.runcall(trainer_train)
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

            cv2.imshow('hi',neuron_syns)
            cv2.waitKey(10)


if __name__ == '__main__':
    check_distance(5,dist_fac=2, size=(20, 10))
