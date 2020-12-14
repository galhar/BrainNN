# Writer: Gal Harari
# Date: 14/12/2020
from font_prediction import FontDataLoader
from brainNN import BrainNN
from hooks import ClassesEvalHook, SaveByEvalHook
from utils.train_utils import DefaultOptimizer, Trainer

train_dir = 'data/Font images/Calibri Font images/'

def create_trainer(epoches=17):
    nodes_details = []
    IINs_details = [(3, 3), (3, 3), (1, 1)]
    inter_connections = [(False, True), (True, True), (True, True)]
    configuration_args = {BrainNN.NODES_DETAILS: nodes_details,
                          BrainNN.IINS_PER_LAYER_NUM: IINs_details,
                          BrainNN.INTER_CONNECTIONS_PER_LAYER: inter_connections}

    net = BrainNN(configuration_args)
    data_loader = FontDataLoader(train_dir)
    optimizer = DefaultOptimizer(net=net, epoches=epoches, sample_reps=6)
    trainer = Trainer(net, data_loader, optimizer, verbose=False)
    return net, trainer


def trainer_evaluation(epoches=20):
    net, trainer = create_trainer(epoches)
    trainer.register_hook(lambda trainer: ClassesEvalHook(trainer, FontDataLoader(
        batched=True)))
    trainer.register_hook(lambda trainer: SaveByEvalHook(trainer, req_acc=70))
    trainer.train()
    tot_acc_str, cls_acc_str = ClassesEvalHook.TOT_ACC_STR, ClassesEvalHook.CLS_ACC_STR
    return [trainer.storage[tot_acc_str]]


if __name__ == '__main__':
    print(trainer_evaluation(epoches=2))