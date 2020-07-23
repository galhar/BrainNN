# Writer: Gal Harari
# Date: 04/06/2020
import numpy as np
import timeit
import cv2
import imutils


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


def test_imshow_size():
    window_name = 'frame'
    frame = np.zeros((700, 1500, 3))
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)

    cv2.imshow(window_name, frame)
    cv2.waitKey()


def test_imshow_color():
    color = (0.1,0.21,0)
    made_frame = color* np.ones((1000, 1000, 3))
    img_frame = cv2.imread("test_image.jpg")
    img_frame = imutils.resize(img_frame, width=100)

    cv2.imshow('image frame', img_frame)
    cv2.imshow('made frame', made_frame)
    cv2.waitKey()


if __name__ == '__main__':
    test_imshow_color()
