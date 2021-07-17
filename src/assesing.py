# Writer: Gal Harari
# Date: 17/11/2020
from src.binary_encoding_task.binary_prediction import create_binary_input_generator, N, \
    evaluate_binary_representation_nn
from src.binary_encoding_task.main_binary import trainer_evaluation, \
    one_one_evaluation, output_distribution_query
from src.Identity_task.main_identity import identity_evaluation
from src.Fonts_task.main_font import fonts_trainer_evaluation, mnist_train_evaluate, \
    mnist_output_dist
from src.utils.general_utils import save_json, load_json, lighten_color, update_prop

import numpy as np
import time
from tqdm import tqdm

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

DATA_PATH = '../records/'

DISABLE_SCIVIEW = True

if DISABLE_SCIVIEW:
    import matplotlib

    matplotlib.use('TkAgg')


def dim(a):
    if not (type(a) == list or type(a) == np.ndarray):
        return []
    # In case of empty list:
    if len(a) == 0:
        return []
    return [len(a)] + dim(a[0])


def get_strings(arr, large_str=""):
    """

    :param arr: largest available arr := [[[str_i,data_i_iter_j],...(i++)],...(j++)]
    :param large_str:
    :return:
    """
    if type(arr[0]) == str:
        return arr[0]
    if type(arr[0]) == list:
        if type(arr[0][0]) == list:
            for i in range(len(arr[0])):
                # here it means arr[0] := [[str_i,data_i_iter_j],...(i++)]
                addon = get_strings(arr[0][i], large_str)
                large_str += (" " + addon) if addon else ""
            return large_str

        for i in range(len(arr)):
            addon = get_strings(arr[i], large_str)
            large_str += (" " + addon) if addon else ""
        return large_str
    return ""


def check_architecture():
    return [trainer_train(), script_training()]


setattr(check_architecture, 'title', 'Regular vs new OOP')
setattr(check_architecture, 'x_label', 'OOP, Regular')
setattr(check_architecture, 'y_label', 'Accuracy')


def average_over_nets(net_data_func_with_attr, iterations=20, is_3D=False,
                      load=None, scatter=True):
    """
    :param net_data_func_with_attr: a function that returns records vector. May return a
    2D records matrix and then set is_3d to True. May return a list of records vectors,
    that some of them can be 3D, it will automatically recognize it.
    :param iterations: how many iteration to do with the net_data_func
    :param load: load_path as a string, or the data itself to process as an array
    :return:
    """
    titles = {}
    if hasattr(net_data_func_with_attr, 'title'):
        titles['title'] = net_data_func_with_attr.title
    else:
        titles['title'] = ''
    titles['title'] += ' '

    if hasattr(net_data_func_with_attr, 'x_label'):
        titles['x_label'] = net_data_func_with_attr.x_label
    if hasattr(net_data_func_with_attr, 'y_label'):
        titles['y_label'] = net_data_func_with_attr.y_label
    if hasattr(net_data_func_with_attr, 'z_label'):
        titles['z_label'] = net_data_func_with_attr.z_label

    if not load:
        iterations_vec = []
        for i in tqdm(range(iterations)):
            iterations_vec.append(net_data_func_with_attr())
            # print(f"Assessing func: {i + 1}\\{iterations}")

        current_time = timestamp()
        save_json(iterations_vec, DATA_PATH + f"{titles.get('title', '')} records"
                                              f"{get_strings(iterations_vec)}"
                                              f" {current_time}")
    elif isinstance(load, str):
        # Here it means we got the data path to load from
        iterations_vec = load_json(DATA_PATH + load)
    else:
        # Here it means we got the data as an array
        iterations_vec = load
        current_time = timestamp()
        save_json(iterations_vec, DATA_PATH + f"{titles.get('title', '')} records"
                                              f"{get_strings(iterations_vec)}"
                                              f" {current_time}")

    # If it's data for only 1 plot, with str documentation [[<title>, <iter_1_data>],
    # [<title>, <iter_2_data>],...]
    if type(iterations_vec[0][0]) == str:
        titles_copy = titles.copy()
        titles_copy['title'] += iterations_vec[0][0]
        describe(titles_copy, [data for s, data in iterations_vec], scatter)

    if not is_3D and (type(iterations_vec[0][0]) == list or type(iterations_vec[0][0])
                      == np.ndarray):
        # Here it means the function returns some vecs of records in one list. Thus
        # iteration_vec looks like: [<some_vecs_of_records_iter_1>,
        # <some_vecs_of_records_iter_2>,...]
        for i in range(len(iterations_vec[0])):
            # If it got title in it take it and continue
            if type(iterations_vec[0][i][0]) == str:
                # <some_vecs_of_records_iter_i> :=[[<title1>, <iter_i_data1>],[<title2>,
                # <iter_i_data2>],...]
                titles_copy = titles.copy()
                titles_copy['title'] += iterations_vec[0][i][0]
                describe(titles_copy, [net_data[i][1] for net_data in iterations_vec],
                         scatter)
            else:
                describe(titles, [net_data[i] for net_data in iterations_vec], scatter)
    else:
        describe(titles, iterations_vec, scatter)


def timestamp():
    current_time = time.strftime("%m_%d_%y %H_%M", time.localtime())
    return current_time


def describe(titles, iterations_vec, scatter):
    """
    plots the records of diffrenet runs, averaging over it
    :param titles: {'title':<graph title>,'x_label':<x_label> ,'y_label':<y_label> ,
    'z_label':<z_label>
    :param iterations_vec: list of the same records collected from different runs,
    single run's
    records may be 2d array or 1d array: [<data_from_run_1>,<data_from_run_2>,...]
    :return:
    """
    iterations = len(iterations_vec)
    is_3D = False
    # Automatically identify is it is 3d. The assumption is it's either an array of 2D
    # mats (require a 3D plot) of an array of 1D arrays (requires a regular 2D plot)
    if len(dim(iterations_vec)) > 2:
        is_3D = True
        # Cancel 3d plots -
        return

    if is_3D:
        x = np.arange(len(iterations_vec[0][0]))
        y = np.arange(len(iterations_vec[0]))
        x, y = np.meshgrid(x, y)
    else:
        x = np.arange(len(iterations_vec[0]))
    avg = np.mean(iterations_vec, axis=0)
    std = np.std(iterations_vec, axis=0)

    # Get exact numbers in avg for max and min
    print_min_max(avg, is_3D, x)

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
        # ax.pcolormesh(x, y, avg.reshape(x.shape))
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
        ax.errorbar(x, avg, ecolor='k',yerr=std, capthick=1, capsize=3)
        plt.xticks(x, [l + 1 for l in x])
        plt.annotate('%0.2f' % avg[-1], xy=(x[-1], avg[-1]), xytext=(3, -3),
                     xycoords='data', textcoords='offset points')
        plt.annotate('%0.2f' % avg[0], xy=(x[0], avg[0]), xytext=(-30, 0),
                     xycoords='data', textcoords='offset points')

    ax.set_title(titles.get('title', ''))
    ax.set_xlabel(titles.get('x_label', ''))
    ax.set_ylabel(titles.get('y_label', ''))
    if is_3D:
        ax.set_zlabel(titles.get('z_label', ''))

    plt.show()

    # current_time = timestamp()
    # fig.savefig(DATA_PATH + f'records {current_time}.png')


def print_min_max(avg, is_3D, x):
    if not avg.any():
        # In case of empty list
        return

    max_idx, min_idx = np.argmax(avg), np.argmin(avg)
    if is_3D:
        max_idx = np.unravel_index(max_idx, avg.shape)
        min_idx = np.unravel_index(min_idx, avg.shape)
    print(f"Max: ({x[max_idx]},{avg[max_idx]}) | Min: ({x[min_idx]},{avg[min_idx]}) | "
          f"3D={is_3D}")


def dummy_func():
    return np.random.normal(1, 0.5, (10,))


def dummy_3d_func():
    return np.reshape(np.arange(13 * 5), (13, 5))


def combine_plots(load_files, labels, colors, title="", x_label="", y_label=""):
    plots_data = []
    for file_name in load_files:
        iterations_vec = load_json(DATA_PATH + file_name)

        # the function returns some vecs of records in one list. Thus
        # iteration_vec looks like: [<some_vecs_of_records_iter_1>,
        # <some_vecs_of_records_iter_2>,...]
        for i in range(len(iterations_vec[0])):
            cur_data = [net_data[i] for net_data in iterations_vec]
            if len(dim(cur_data)) > 2:
                # Cancel 3d plots -
                continue
            plots_data.append(cur_data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, plt_data in enumerate(plots_data):
        x = np.arange(len(plt_data[0]))
        avg = np.mean(plt_data, axis=0)
        std = np.std(plt_data, axis=0)
        color = colors[i]

        plt.plot(x, avg, label=labels[i], color=color, marker='o', markersize=3)
        std_edge_c, std_reg_c = lighten_color(color, 0.3), lighten_color(color, 0.05)
        plt.fill_between(x, avg - std, avg + std, facecolor=std_reg_c, alpha=1.0,
                         edgecolor=std_edge_c, linewidth=1, linestyle='dashed')

        # ax.errorbar(x, avg, ecolor='k', yerr=std, capthick=1, capsize=3, label=labels[
        #     i], c=colors[i])
        plt.xticks(x, [l + 1 for l in x])
        plt.annotate('%0.2f' % avg[-1], xy=(x[-1], avg[-1]), xytext=(3, -5 + (1-i)*5),
                     xycoords='data', textcoords='offset points', c=colors[i])
        plt.annotate('%0.2f' % avg[0], xy=(x[0], avg[0]), xytext=(-30, -5 + (1-i)*5),
                     xycoords='data', textcoords='offset points', c=colors[i])

    # Remove errorbars from the legend
    # get handles
    # handles, labels = ax.get_legend_handles_labels()
    # # remove the errorbars
    # handles = [h[0] for h in handles]
    # # use them in the legend
    # ax.legend(handles, labels, loc='upper right', numpoints=1)

    if len(plots_data) > 1:
        plt.legend(handler_map={plt.Line2D: HandlerLine2D(update_func=update_prop)},
                   loc='upper right')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()





if __name__ == '__main__':
    # load_path = 'Fonts accuracy over 10 nets, 2 layers, with decrease  records ' \
    #             '07_07_21 02_10.json'
    check_func = fonts_trainer_evaluation
    setattr(check_func, 'title', 'Fonts accuracy over 10 nets, 3 layers, 100 hidden '
                                 'winners, NO SKIP connections')
    setattr(check_func, 'x_label', 'Epochs')
    setattr(check_func, 'y_label', 'Accuracy')
    average_over_nets(check_func, iterations=10)#, scatter=False, load=load_path)
    # data_files = [
    #     "Fonts accuracy over 10 nets, 3 layers, 100 hidden winners  records 07_15_21 23_58.json",
    #     "Fonts accuracy over 10 nets, 2 layers, low sample rep  records 07_14_21 "
    #     "00_26.json",
    #     # "Fonts accuracy over 10 nets, 2 layers 30 ep  records 07_09_21 01_28.json"
    #
    # ]
    # labels = [
    #     "3 layers, 6 output spikes",
    #     "2 layers, 6 output spikes",
    #     "2 layers, 11 output spikes",
    # ]
    # title = "Instant Model Accuracy On Fonts Learnability"
    # combine_plots(data_files,labels,['blue','green', 'purple'],title, "Epochs",
    #               "Accuracy")
