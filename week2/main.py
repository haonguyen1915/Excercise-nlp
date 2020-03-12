# https://colab.research.google.com/drive/1zV0IEDv9t66HeEOMfBVJzRlys5vfDJNe
from week2.utils.setup_google_colab import setup_week2
from haolib import *
from week2.utils.utility import *
from week2.utils.dataset import *
from week2.utils.model import SurnameGenerationModel
from torch import optim
from haolib.lib_ai.learner import Learner
prj_dir = get_cfd(backward=1)


def set_up():
    setup_week2()


def download_dataset():
    import sys
    sys.path.append("..")
    from week2.common.download_utils import download_week2_resources
    download_week2_resources()


def get_data(bs):
    words_vocab, tags_vob = build_vocabs()
    my_vectorizer = MyVectorizer(words_vocab, tags_vob)
    train_tokens, train_tags = read_data('{}/week2/data/train.txt'.format(prj_dir))
    validation_tokens, validation_tags = read_data('{}/week2/data/validation.txt'.format(prj_dir))
    test_tokens, test_tags = read_data('{}/week2/data/test.txt'.format(prj_dir))

    train_ds = MyDataset(my_vectorizer, train_tokens, train_tags)
    valid_ds = MyDataset(my_vectorizer, validation_tokens, validation_tags)
    test_ds = MyDataset(my_vectorizer, test_tokens, test_tags)

    data_container = DataContainer(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, bs=bs)
    print("mask_index: {}".format(my_vectorizer.words_vocab.mask_index))
    print("unk_index: {}".format(my_vectorizer.words_vocab.unk_index))

    return data_container, my_vectorizer


def get_learner():
    # Model hyper parameters
    char_embedding_size = 32
    rnn_hidden_size = 32
    # Training hyper parameter
    learning_rate = 1e-3
    batch_size = 128
    surname_container, vectorizer = get_data(bs=batch_size)
    print(surname_container)
    model = SurnameGenerationModel(char_embedding_size=char_embedding_size,
                                   char_vocab_size=len(vectorizer.words_vocab),
                                   rnn_hidden_size=rnn_hidden_size,
                                   padding_idx=vectorizer.words_vocab.mask_index)
    loss_func = sequence_loss
    # optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)
    learner = Learner(model=model, data_container=surname_container, loss_func=loss_func, opt_func=optimizer,
                      acc_func=compute_accuracy)
    return learner


def train():
    learner = get_learner()
    learner.model_info()
    # # learner.lr_finder()
    # # learner.recoder.plot()
    try:
        learner.load("{}/week2/data/statge01".format(prj_dir))
    except FileNotFoundError:
        learner.fit_one_cycle(2, 0.001, only_save_best=True)
        learner.save("{}/week2/data/statge01".format(prj_dir))
        learner.export("{}/week2/data/model.pth".format(prj_dir))


def evaluation():
    surname_container, dataset = get_data(bs=64)
    vectorizer = dataset.get_vectorizer()
    learner = get_learner()
    learner.load("{}/week2/data/statge01".format(prj_dir))
    # number of names to generate
    num_names = 10
    model = load_model("{}/week2/data/model.pth".format(prj_dir))
    model.to("cpu")
    # Generate nationality hidden state
    sampled_surnames = decode_samples(
        sample_from_model(model, vectorizer, num_samples=num_names),
        vectorizer)
    # Show results
    print("-" * 15)
    for i in range(num_names):
        print(sampled_surnames[i])


if __name__ == "__main__":
    train()
    # evaluation()
    # predict("Sarraf")
