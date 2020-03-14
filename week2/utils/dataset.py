from week2.utils.utility import *
from week2.utils.constants import *
from haolib import *
from haolib.lib_nlp.vocabulary import SequenceVocabulary, Vocabulary
from haolib.lib_nlp.vectorizer import Vectorizer
from haolib.lib_nlp.text_container import TextContainer

prj_dir = get_cfd(backward=2)


def understanding_data():
    train_tokens, train_tags = read_data('{}/week2/data/train.txt'.format(prj_dir))
    # validation_tokens, validation_tags = read_data('data/validation.txt')
    # test_tokens, test_tags = read_data('data/test.txt')
    for i in range(3):
        for token, tag in zip(train_tokens[i], train_tags[i]):
            print('%s\t%s' % (token, tag))
        print()


def build_vocabs(train_tokens, train_tags, validation_tokens, begin_seq_token=None, end_seq_token=None):
    token2idx, idx2token = build_dict(train_tokens + validation_tokens)
    tag2idx, idx2tag = build_dict(train_tags)
    words_vocab = SequenceVocabulary(token_to_idx=token2idx, unk_token='<UNK>', mask_token='<PAD>')
    if begin_seq_token is not None and end_seq_token is not None:
        tags_vob = SequenceVocabulary(token_to_idx=tag2idx, begin_seq_token=begin_seq_token, end_seq_token=end_seq_token)
    else:
        tags_vob = Vocabulary(token_to_idx=tag2idx)
    return words_vocab, tags_vob


class MyVectorizer(Vectorizer):
    def __init__(self, words_vob, tags_vob):
        super().__init__(words_vob, tags_vob)

    def vectorize(self, text, vector_length=-1):
        """
        param: text(list): List of token
        param: vector_length(int): max len of seq
        """
        indices = []
        indices.extend(self.words_vocab.lookup_token(token)
                       for token in text)
        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.words_vocab.mask_index

        return out_vector, len(indices)


class MyDataset(BaseDataset):
    def __init__(self, vectorizer, data, targets, max_seq_length=50):
        """

        """
        super().__init__()
        self.vectorizer = vectorizer
        self.data = data
        self.targets = targets
        self.classes = [k for k, _ in vectorizer.tags_vocab.to_serializable()["token_to_idx"].items()]
        self.class_to_idx = vectorizer.tags_vocab.to_serializable()["token_to_idx"]
        self._max_seq_length = max_seq_length

    def __len(self):
        return len(data)

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

                Args:
                    index (int): the index to the data point
                Returns:
                    a dictionary holding the data point's:
                        features (x_data)
                        label (y_target)
                        feature length (x_length)
                """

        word_vector, vec_length = self.vectorizer.vectorize(self.data[index], -1)

        tag_indices = []
        tag_indices.extend(self.vectorizer.tags_vocab.lookup_token(token)
                           for token in self.targets[index])
        tag_vector = np.zeros(len(word_vector), dtype=np.int64)
        tag_vector[:len(tag_indices)] = tag_indices
        tag_vector[len(tag_indices):] = self.vectorizer.tags_vocab.lookup_token('O')
        # tag_index = self.vectorizer.tags_vocab.lookup_token(index)
        return word_vector, tag_vector


def get_toy_data_vecterizer(bs=1, tart_seq_token=None, end_seq_token=None):
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]
    train_tokens = [words[0] for words in training_data]
    train_tags = [words[1] for words in training_data]

    words_vocab, tags_vob = build_vocabs(train_tokens, train_tags, train_tokens, tart_seq_token, end_seq_token)
    my_vectorizer = MyVectorizer(words_vocab, tags_vob)
    train_ds = MyDataset(my_vectorizer, train_tokens, train_tags)
    valid_ds = MyDataset(my_vectorizer, train_tokens, train_tags)
    test_ds = MyDataset(my_vectorizer, train_tokens, train_tags)
    data_container = TextContainer(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, bs=bs, vectorizer=my_vectorizer)

    return data_container, my_vectorizer


def get_data_vecterizer(bs, vectorizer_path="./vecterizer_default.pkl", start_seq_token=None, end_seq_token=None):
    train_tokens, train_tags = read_data('{}/week2/data/train.txt'.format(prj_dir))
    validation_tokens, validation_tags = read_data('{}/week2/data/validation.txt'.format(prj_dir))
    test_tokens, test_tags = read_data('{}/week2/data/test.txt'.format(prj_dir))

    words_vocab, tags_vob = build_vocabs(train_tokens, train_tags, validation_tokens, start_seq_token, end_seq_token)
    if vectorizer_path is not None and os.path.isfile(vectorizer_path):
        my_vectorizer = load_context(vectorizer_path)
        print("Loaded vectorizer succsesfully")
    else:
        my_vectorizer = MyVectorizer(words_vocab, tags_vob)
        save_context(my_vectorizer, vectorizer_path)
    train_ds = MyDataset(my_vectorizer, train_tokens, train_tags)
    valid_ds = MyDataset(my_vectorizer, validation_tokens, validation_tags)
    test_ds = MyDataset(my_vectorizer, test_tokens, test_tags)

    data_container = TextContainer(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, bs=bs)

    return data_container, my_vectorizer


def test_dataset():
    words_vocab, tags_vob = build_vocabs()
    my_vectorizer = MyVectorizer(words_vocab, tags_vob)
    text = ['RT', '<USR>', ':', 'Online', 'ticket', 'sales', 'for', 'Ghostland', 'Observatory', 'extended', 'until',
            '6', 'PM', 'EST', 'due', 'to', 'high', 'demand', '.', 'Get', 'them', 'before', 'they', 'sell', 'out', '...']

    out_vector, len_ = my_vectorizer.vectorize(text)
    print("out_vector: {}".format(out_vector))
    print("len_: {}".format(len_))
    print(my_vectorizer.words_vocab.to_serializable()["token_to_idx"])
    # classes = [k for k, _ in my_vectorizer.words_vocab.to_serializable()["token_to_idx"].items()]
    # print(classes)
    # exit()

    train_tokens, train_tags = read_data('{}/week2/data/train.txt'.format(prj_dir))
    validation_tokens, validation_tags = read_data('{}/week2/data/validation.txt'.format(prj_dir))
    test_tokens, test_tags = read_data('{}/week2/data/test.txt'.format(prj_dir))

    my_dataset = MyDataset(my_vectorizer, train_tokens, train_tags)
    show_dict(my_dataset.classes)
    show_dict(my_dataset.class_to_idx)

    x, y = my_dataset[0]
    print(x)
    print(y)


def test_data_container():
    words_vocab, tags_vob = build_vocabs()
    my_vectorizer = MyVectorizer(words_vocab, tags_vob)
    train_tokens, train_tags = read_data('{}/week2/data/train.txt'.format(prj_dir))
    validation_tokens, validation_tags = read_data('{}/week2/data/validation.txt'.format(prj_dir))
    test_tokens, test_tags = read_data('{}/week2/data/test.txt'.format(prj_dir))
    train_ds = MyDataset(my_vectorizer, train_tokens, train_tags)
    valid_ds = MyDataset(my_vectorizer, validation_tokens, validation_tags)
    test_ds = MyDataset(my_vectorizer, test_tokens, test_tags)

    data_container = TextContainer(train_ds=train_ds, valid_ds=valid_ds, bs=4)

    train_loader = data_container.get_train_dl()
    for data in train_loader:
        temple_data = data
        break
    x, y = temple_data
    describe_tensor(x)
    describe_tensor(y)


if __name__ == "__main__":
    # understanding_data()
    test_data_container()
