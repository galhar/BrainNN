# Writer: Gal Harari
# Date: 17/11/2020
from binary_prediction import create_binary_input_generator, N, \
    evaluate_binary_representation_nn
from brainNN import BrainNN
import numpy as np
import pickle
import time

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from main import trainer_train, script_training, trainer_evaluation

DATA_PATH = 'data/'


def save(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def dim(a):
    if not (type(a) == list or type(a) == np.ndarray):
        return []
    return [len(a)] + dim(a[0])


def check_architecture():
    return [trainer_train(), script_training()]


setattr(check_architecture, 'title', 'Regular vs new OOP')
setattr(check_architecture, 'x_label', 'OOP, Regular')
setattr(check_architecture, 'y_label', 'Accuracy')


def average_over_nets(net_data_func_with_attr, iterations=20, is_3D=False,
                      load_path=None, scatter=True):
    """
    :param net_data_func_with_attr: a function that returns data vector. May return a
    2D data matrix and then set is_3d to True. May return a list of data vectors,
    that some of them can be 3D, it will automatically recognize it.
    :param iterations: how many iteration to do with the net_data_func
    :return:
    """
    if not load_path:
        iterations_vec = []
        for i in range(iterations):
            iterations_vec.append(net_data_func_with_attr())
            print(f"Assessing func: {i + 1}\\{iterations}")

        current_time = timestamp()
        save(DATA_PATH + f'Pickle data {current_time}', iterations_vec)
    else:
        with open(DATA_PATH + load_path, "rb") as input_file:
            iterations_vec = pickle.load(input_file)

    titles = {}
    if hasattr(net_data_func_with_attr, 'title'):
        titles['title'] = net_data_func_with_attr.title
    if hasattr(net_data_func_with_attr, 'x_label'):
        titles['x_label'] = net_data_func_with_attr.x_label
    if hasattr(net_data_func_with_attr, 'y_label'):
        titles['y_label'] = net_data_func_with_attr.y_label
    if hasattr(net_data_func_with_attr, 'z_label'):
        titles['z_label'] = net_data_func_with_attr.z_label

    if not is_3D and (type(iterations_vec[0][0]) == list or type(iterations_vec[0][0])
                      == np.ndarray):
        # Here it means the function returns some vecs of data inone list.
        for i in range(len(iterations_vec[0])):
            describe(titles, [net_data[i] for net_data in iterations_vec], scatter)
    else:
        describe(titles, iterations_vec, scatter)


def timestamp():
    current_time = time.strftime("%m_%d_%y %H_%M", time.localtime())
    return current_time


def describe(titles, iterations_vec, scatter):
    """
    plots the data of diffrenet runs, averaging over it
    :param titles: {'title':<graph title>,'x_label':<x_label> ,'y_label':<y_label> ,
    'z_label':<z_label>
    :param iterations_vec: list of the same data collected from different runs,
    single run's
    data may be 2d array or 1d array: [<data_from_run_1>,<data_from_run_2>,...]
    :return:
    """
    iterations = len(iterations_vec)
    is_3D = False
    # Automatically identify is it is 3d. The assumption is it's either an array of 2D
    # mats (require a 3D plot) of an array of 1D arrays (requires a regular 2D plot)
    if len(dim(iterations_vec)) > 2:
        is_3D = True

    if is_3D:
        x = np.arange(len(iterations_vec[0][0]))
        y = np.arange(len(iterations_vec[0]))
        x, y = np.meshgrid(x, y)
    else:
        x = [i for i in range(len(iterations_vec[0]))]
    avg = np.mean(iterations_vec, axis=0)
    std = np.std(iterations_vec, axis=0)

    # Plot part:
    fig = plt.figure()
    if is_3D:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # plot all the runs scattered
    if scatter:
        if is_3D:
            for i in range(iterations):
                ax.scatter(x, y, iterations_vec[i])
        else:
            for i in range(iterations):
                ax.scatter(x, iterations_vec[i])
    # plot average and std
    if is_3D:
        # average:
        surf = ax.plot_surface(x, y, avg, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # std:
        x_len, y_len = avg.shape
        for i in range(x_len):
            for j in range(y_len):
                ax.plot([x[i][j], x[i][j]], [y[i][j], y[i][j]],
                        [avg[i][j] + std[i][j], avg[i][j] - std[i][j]], marker="_")
    else:
        # average + std:
        ax.errorbar(x, avg, std)

    ax.set_title(titles.get('title', ''))
    ax.set_xlabel(titles.get('x_label', ''))
    ax.set_ylabel(titles.get('y_label', ''))
    if is_3D:
        ax.set_zlabel(titles.get('z_label', ''))

    plt.show()

    current_time = timestamp()
    fig.savefig(DATA_PATH + f'data {current_time}.png')


def dummy_func():
    return np.random.normal(1, 0.5, (10,))


def dummy_3d_func():
    return np.reshape(np.arange(13 * 5), (13, 5))


if __name__ == '__main__':
    # average_one_checks(dummy_3d_func, runs=2)
    average_over_nets(trainer_evaluation, iterations=60)
