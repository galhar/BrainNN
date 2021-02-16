# Writer: Gal Harari
# Date: 18/11/2020
import weakref
from tqdm import tqdm
from src.utils.train_utils import EvalNetWrapper
import numpy as np


class HookBase:

    def __init__(self, trainer):
        self._trainer = weakref.proxy(trainer)


    def after_batch(self):
        """
        A function called by the trainer after each batch. The trainer zeros the
        neurons after every batch.
        :return:
        """
        pass


    def after_epoch(self):
        """
        A function called by the trainer after each epoch
        :return:
        """
        pass


class SaveHook(HookBase):

    def __init__(self, trainer, save_name=None, save_after=5, overwrite=True):
        """

        :param trainer:
        :param save_name: name to save as without the extension
        :param save_after: save after this number of epoches, repeatedly
        :param overwrite: if another file with the same name exists, do you want to
        overwrite it.
        """
        super(SaveHook, self).__init__(trainer)
        self._save_name = save_name if save_name else 'NetSavedByHook'
        self._save_after_epoches = save_after
        self._overwrite = overwrite
        self._net = trainer.net
        self._epoches_counter = 0


    def after_epoch(self):
        self._epoches_counter = (self._epoches_counter + 1) % self._save_after_epoches
        if self._epoches_counter == 0:
            self._net.save_state(name=self._save_name, overwrite=self._overwrite)


class SaveByEvalHook(HookBase):
    """
    Always register after the eval hook is registered
    """


    def __init__(self, trainer, req_acc=70, save_name=None, overwrite=True):
        """

        :param trainer:
        :param save_name: name to save as without the extension
        :param save_after: save after this number of epoches, repeatedly
        :param overwrite: if another file with the same name exists, do you want to
        overwrite it.
        """
        super().__init__(trainer)
        self._save_name = save_name if save_name else ('NetSavedByHook acc %s' % req_acc)
        self._overwrite = overwrite
        self._net = trainer.net
        self._req_acc = req_acc


    def after_epoch(self):
        current_acc = self._trainer.storage[ClassesEvalHook.TOT_ACC_STR][-1]
        if current_acc >= self._req_acc:
            self._net.save_state(name=self._save_name, overwrite=self._overwrite)


class ClassesEvalHook(HookBase):
    CLS_ACC_STR = 'Classes accuracy over epochs'
    TOT_ACC_STR = 'Total accuracy over epochs'


    def __init__(self, trainer, data_loader, req_shots_num=5, noise_std=0,
                 vis_last_ep=False):
        """

        :param trainer:
        :param data_loader: In order to zero between checks, put each sample in a
        different batch. It zeros the neurons between batches.
        :param req_shots_num:
        :param noise_std:
        """
        super(ClassesEvalHook, self).__init__(trainer)
        self._ep_idx = 0

        self._req_shots_n = req_shots_num
        self._net = trainer.net
        self._data_loader = data_loader
        self._net_wrapper = EvalNetWrapper(self._net,
                                           noise_std=noise_std,
                                           req_shots_num=req_shots_num)

        self._vis_last_ep = vis_last_ep
        self._last_epoch = self._trainer.optimizer.epochs
        self._verbose = self._trainer._verbose

        self._cls_lst = self._data_loader.classes_neurons
        self._trainer.storage[ClassesEvalHook.CLS_ACC_STR] = []
        self._trainer.storage[ClassesEvalHook.TOT_ACC_STR] = []


    def after_epoch(self):
        if self._verbose:
            print("[*] Measuring Accuracy...")
        self._net.freeze()
        class_correct = np.zeros_like(self._cls_lst)
        class_total = np.zeros_like(self._cls_lst)
        for sample_batch, labels in self._data_loader:
            samples_idxs = range(len(sample_batch))
            if self._verbose and not self._data_loader._batched:
                samples_idxs = tqdm(samples_idxs)

            for i in samples_idxs:
                sample, l = sample_batch[i], labels[i]

                if self._verbose:
                    # Counting on that the loader is of type ClassesDataLoader
                    print(self._data_loader.neuron_to_class_dict[l])

                output = self._net_wrapper(sample)
                pred_y = np.argmax(output)

                class_correct[l] += 1 if pred_y == l else 0
                class_total[l] += 1
                self._net.zero_neurons()

        classes_acc = 100 * class_correct / class_total
        mean_acc = np.mean(classes_acc)

        # Save to trainer history
        self._trainer.storage[ClassesEvalHook.CLS_ACC_STR].append(classes_acc)
        self._trainer.storage[ClassesEvalHook.TOT_ACC_STR].append(mean_acc)
        print("[*] Mean accuracy =", mean_acc)
        self._ep_idx += 1
        self._net.unfreeze()

        if self._ep_idx == self._last_epoch and self._vis_last_ep:
            self.visualize()


    def visualize(self):
        self._net.freeze()
        self._net.set_visualization('debug')

        for sample_batch, labels in self._data_loader:
            for i in range(len(sample_batch)):
                sample, l = sample_batch[i], labels[i]
                output = self._net_wrapper(sample)
                pred_y = np.argmax(output)

                print("Output:", list(output), "|", pred_y)

                self._net.zero_neurons()

        self._net.unfreeze()


class OutputDistributionHook(HookBase):
    CLS_DIST_STR = 'Classes output distribution over epochs'


    def __init__(self, trainer, data_loader, neurons_to_check=None, req_shots_num=5,
                 noise_std=0):
        """

        :param trainer:
        :param data_loader: In order to zero between checks, put each sample in a
        different batch. It zeros the neurons between batches.
        :param req_shots_num:
        :param noise_std:
        """
        super().__init__(trainer)
        self._interest_labels = neurons_to_check if neurons_to_check else \
            data_loader.classes_neurons

        self._net = trainer.net
        self._data_loader = data_loader
        self._net_wrapper = EvalNetWrapper(self._net,
                                           noise_std=noise_std,
                                           req_shots_num=req_shots_num)
        self._cls_lst = self._data_loader.classes_neurons

        self._storage_dict = {}
        for label in self._interest_labels:
            self._storage_dict[label] = []
        self._trainer.storage[OutputDistributionHook.CLS_DIST_STR] = self._storage_dict


    def after_epoch(self):
        self._net.freeze()

        for sample_batch, labels in self._data_loader:
            for i in range(len(sample_batch)):
                sample, l = sample_batch[i], labels[i]

                if l not in self._interest_labels:
                    continue

                output = self._net_wrapper(sample)
                self._storage_dict[l].append(output)

                self._net.zero_neurons()

        self._net.unfreeze()
