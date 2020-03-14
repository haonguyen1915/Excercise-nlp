# https://colab.research.google.com/drive/1zV0IEDv9t66HeEOMfBVJzRlys5vfDJNe
# Mine: https://colab.research.google.com/drive/1bDT_yMj1fnHMFG9wo-mmeBahPO5aqYXJ#scrollTo=hr5PShABrXTV
from week2.utils.setup_google_colab import setup_week2
from haolib import *
from week2.utils.utility import *
from week2.utils.dataset import *
from week2.utils.model import LSTMNERClassifier, BiLSTM_CRF
from torch import optim
from haolib.lib_nlp.seq2seq_learner import Seq2SeqLearner
from torch import nn

prj_dir = get_cfd(backward=1)
vectorizer_path = '{}/week2/data/vecterizer_02.pkl'.format(prj_dir)
model_path = '{}/week2/data/model_02.pkl'.format(prj_dir)

seed_everything(777)
# Model hyper parameters
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
# Training hyper parameter
learning_rate = 1e-3


def get_learner(data_container, vectorizer):
    model = BiLSTM_CRF(len(vectorizer.words_vocab), vectorizer.tags_vocab.token_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
    loss_func = model.neg_log_likelihood
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)
    learner = Seq2SeqLearner(model=model, data_container=data_container, loss_func=loss_func, opt_func=optimizer,
                             no_batch=True, mask_idx_acc=-1)
    learner.custome_loss(loss_func, LOSS_WITH_INPUT)

    return learner

def get_model(data_container, vectorizer):
    model = BiLSTM_CRF(len(vectorizer.words_vocab), vectorizer.tags_vocab.token_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
    loss_func = model.neg_log_likelihood
    return model, loss_func

# def train_toy():
#     data_container, vectorizer = get_toy_data_vecterizer(bs=1)
#     model, loss_func = get_model(data_container, vectorizer)
#     optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
#     iterator = iter(data_container.get_train_dl())
#     with torch.no_grad():
#         # precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#         # precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
#         X, y = iterator.__next__()
#         X = torch.squeeze(X, dim=0)
#         y = torch.squeeze(y, dim=0)
#         print("pred: {}".format(model(X)))
#         print("true: {}".format(y))
#
#     # Make sure prepare_sequence from earlier in the LSTM section is loaded
#     for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
#         loss_epoch = 0
#         for sentence, tags in data_container.get_train_dl():
#             # Step 1. Remember that Pytorch accumulates gradients.
#             # We need to clear them out before each instance
#             model.zero_grad()
#
#             # Step 2. Get our inputs ready for the network, that is,
#             # turn them into Tensors of word indices.
#             sentence_in = torch.squeeze(sentence, dim=0)
#             targets = torch.squeeze(tags, dim=0)
#             # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
#             # describe_tensor(targets)
#             # exit()
#
#             # Step 3. Run our forward pass.
#             # loss = model.neg_log_likelihood(sentence_in, targets)
#             loss = loss_func(sentence_in, targets)
#
#             # Step 4. Compute the loss, gradients, and update the parameters by
#             # calling optimizer.step()
#             loss.backward()
#             optimizer.step()
#             loss_epoch = loss
#         # print("loss: {}".format(loss_epoch))
#     iterator = iter(data_container.get_train_dl())
#     with torch.no_grad():
#         X, y = iterator.__next__()
#         X = torch.squeeze(X, dim=0)
#         y = torch.squeeze(y, dim=0)
#         print("pred: {}".format(model(X)))
#         print("true: {}".format(y))
#     torch.save(model, model_path)
#
#
def train_toy():
    data_container, vectorizer = get_toy_data_vecterizer(1, START_TAG, STOP_TAG)
    learner = get_learner(data_container, vectorizer)
    #
    learner.model_info()
    learner.fit_one_cycle(20, 0.1, only_save_best=True)
    learner.plot_confusion_matrix()
def train():
    data_container, vectorizer = get_data_vecterizer(bs=1, vectorizer_path=vectorizer_path)
    learner = get_learner(data_container, vectorizer)

    learner.model_info()
    learner.fit_one_cycle(20, 0.1, only_save_best=True)
    learner.test_n_case(show_indice=True)

if __name__ == "__main__":
    train_toy()
    # train()
    # test_some_case()
    # evaluation()
    # predict("Sarraf")
    # [(tensor([[1., 0., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.]]), tensor([0, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1]), tensor([14.8451])),
    #  (tensor([[1., 0., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [0., 1., 0., 0., 0., 0., 0.],
    #           [1., 0., 0., 0., 0., 0., 0.],
    #           [0., 0., 0., 1., 0., 0., 0.]]), tensor([0, 2, 1, 1, 1, 1, 0]), tensor([7.4639]))]
