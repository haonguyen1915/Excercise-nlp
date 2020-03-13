# https://colab.research.google.com/drive/1zV0IEDv9t66HeEOMfBVJzRlys5vfDJNe
# Mine: https://colab.research.google.com/drive/1bDT_yMj1fnHMFG9wo-mmeBahPO5aqYXJ#scrollTo=hr5PShABrXTV
from week2.utils.setup_google_colab import setup_week2
from haolib import *
from week2.utils.utility import *
from week2.utils.dataset import *
from week2.utils.model import LSTMNERClassifier
from torch import optim
from haolib.lib_ai.seq2seq_learner import Seq2SeqLearner
from torch import nn

prj_dir = get_cfd(backward=1)
vectorizer_path = '{}/week2/data/vecterizer_.pkl'.format(prj_dir)

seed_everything(777)
# Model hyper parameters
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
# Training hyper parameter
learning_rate = 1e-3


def get_learner(data_container, vectorizer):
    loss_func = nn.NLLLoss()
    # acc_func = Acc(mask_idx=0)
    model = LSTMNERClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(vectorizer.words_vocab), len(vectorizer.tags_vocab))
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)
    learner = Seq2SeqLearner(model=model, data_container=data_container, loss_func=loss_func, opt_func=optimizer,
                             no_batch=True, mask_idx_acc=-1)
    return learner


def train_toy():
    data_container, vectorizer = get_toy_data_vecterizer(bs=1)
    learner = get_learner(data_container, vectorizer)
    learner.model_info()
    learner.fit_one_cycle(20, 0.1, only_save_best=True)
    learner.test_n_case(show_indice=True)
    # preds = learner.confusion_matrix()
    preds = learner.plot_confusion_matrix()
    print(learner.validate())
    print(preds)
    # train_loader = learner.data_container.get_train_dl()
    # print("X: {}".format(x[0:1]))
    # print("y:    {}".format(y))
    # out = learner.predict(x[0:1])
    # print("pred: {}".format(out["class_index"]))


def train():
    data_container, vectorizer = get_data_vecterizer(bs=1)
    learner = get_learner(data_container, vectorizer)
    learner.model_info()
    learner.fit_one_cycle(1, 0.1, only_save_best=True)
    learner.test_n_case(show_indice=True)
    # train_loader = learner.data_container.get_train_dl()
    # print("X: {}".format(x[0:1]))
    # print("y:    {}".format(y))
    # out = learner.predict(x[0:1])
    # print("pred: {}".format(out["class_index"]))


def evaluation():
    data_container, vectorizer = get_data_vecterizer(bs=1, vectorizer_path="week2/data/vecterizer_01.pkl")
    learner = get_learner(data_container, vectorizer)
    learner.load("week2/data/model_01.ckpt")
    learner.test_n_case()
    learner.plot_confusion_matrix(ds_type=DS_VALID)

if __name__ == "__main__":
    # train_toy()
    # train()
    # test_some_case()
    evaluation()
    # predict("Sarraf")
