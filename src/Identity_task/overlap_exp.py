# Writer: Gal Harari
# Date: 17/07/2021
DISABLE_SCIVIEW = True
if DISABLE_SCIVIEW:
    import matplotlib

    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
from src.Identity_task.identity_prediction import IdentityDataLoader, N
from src.brainNN import BrainNN
from src.hooks import ClassesEvalHook, ClassesEvalSpikeCountHook
from src.utils.train_utils import DefaultOptimizer, Trainer
from src.utils.general_utils import save_json, load_json, update_prop

SAVE_NAME = 'overlap_data_2_layer.json'
PLT_IDX = 1

def create_trainer(epochs):
    data_loader = IdentityDataLoader()
    inp_len = len(data_loader.samples[0])
    output_shape = len(data_loader.classes_neurons)

    fc = [BrainNN.FC]
    # nodes_details = [inp_len, output_shape * 2, output_shape]
    # IINs_details = [(1,), (1,), (1,)]
    # conn_mat = [[fc, fc, None],
    #             [None, fc, fc],
    #             [None, fc, fc]]
    # winners = [0, 2, 1]
    nodes_details = [inp_len, output_shape]
    IINs_details = [(1, ),  (1, )]
    conn_mat = [[fc, fc],
                [None, fc]]
    winners = [0, 1]

    vis_str = 'None'

    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.CONNECTIONS_MAT: conn_mat,
                          BrainNN.WINNERS_PER_LAYER: winners,
                          BrainNN.VISUALIZATION_FUNC_STR: vis_str}

    net = BrainNN(configuration_args)
    optimizer = DefaultOptimizer(net=net, epochs=epochs, sample_reps=6, sharp=True,
                                 inc_prob=1, dec_prob=0.0)
    trainer = Trainer(net, data_loader, optimizer, verbose=True)
    return net, trainer


def overlap_exp():
    epochs = 5
    net, trainer = create_trainer(epochs)
    trainer.register_hook(
        lambda trainer: ClassesEvalSpikeCountHook(trainer,
                                                  IdentityDataLoader(batched=False,
                                                                     amp=70)))
    trainer.train()
    tot_acc_str = ClassesEvalSpikeCountHook.TOT_ACC_STR
    cls_acc_str = ClassesEvalSpikeCountHook.CLS_ACC_STR
    spike_count_str = ClassesEvalSpikeCountHook.SPIKE_COUNT
    return [trainer.storage[cls_acc_str], trainer.storage[tot_acc_str],
            trainer.storage[spike_count_str]]


def plot_spike_counts(spike_count_dict_over_epochs):
    """
    :param spike_count_dict_over_epochs: [<spike_count_epoch1>, <spike_count_epoch2>,...]
    <spike_count_epoch1> comes as {inp_cls1: [1st neuron spike count, 2nd neurons spike
    count,..., nth neuron spike count], inp_cls2: [1st neuron spike count, 2nd neurons
    spike
    count,..., nth neuron spike count], ...}
    :return:
    """
    cls_n = 9
    x = range(len(list(spike_count_dict_over_epochs[0].values())[0]))
    rows = 3
    cols = 3
    for ep in range(len(spike_count_dict_over_epochs)):
        fig = plt.figure()
        fig.suptitle("Spike Count In Hidden Layer With Non-Overlapped Data")
        for i, dict_pair in enumerate(spike_count_dict_over_epochs[
                                                              ep].items()):
            cls_name, cls_inp_spike_count = dict_pair
            ax = fig.add_subplot(rows, cols, i + 1)
            plt.bar(x, cls_inp_spike_count)
            # plt.xticks(x)
            if i > 5:
                plt.xlabel('Neuron Index In the Hidden Layer')
            if i%3==0:
                plt.ylabel('Spike Count')
            # plt.title('Input %s' % cls_name)
        plt.show()

def plot_acc(load_files, labels, colors, title):
    plots_data = []
    for file_name in load_files:
        plots_data.append(load_json(file_name))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    for i, plt_data in enumerate(plots_data):
        plt_data = plt_data[PLT_IDX]
        x = np.arange(len(plt_data))
        y = plt_data
        color = colors[i]

        plt.plot(x, y, label=labels[i], color=color, marker='o', markersize=3)

        # ax.errorbar(x, avg, ecolor='k', yerr=std, capthick=1, capsize=3, label=labels[
        #     i], c=colors[i])
        plt.xticks(x, [l + 1 for l in x])
        plt.annotate('%0.2f' % y[-1], xy=(x[-1], y[-1]), xytext=(3, -5 + (1 - i) * 5),
                     xycoords='data', textcoords='offset points', c=colors[i])
        plt.annotate('%0.2f' % y[0], xy=(x[0], y[0]), xytext=(-30, -5 + (1 - i) * 5),
                     xycoords='data', textcoords='offset points', c=colors[i])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(handler_map={plt.Line2D: HandlerLine2D(update_func=update_prop)},
               loc='upper right')
    plt.show()
if __name__ == '__main__':
    # save_json(overlap_exp(), SAVE_NAME)
    # cls_acc, tot_acc, spike_count_dict = load_json(SAVE_NAME)
    # plot_spike_counts(spike_count_dict)
    data_files = [
        # "no_overlap_data.json",
        "overlap_data_2_layer.json"
    ]
    labels = [
        "data with overlap",
        "data without overlap",

    ]
    colors = ['green','blue']
    title = "2-Layer Instant Model Accuracy On The Identity Task"
    plot_acc(data_files, labels, colors, title)
