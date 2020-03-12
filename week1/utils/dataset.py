from haolib import *
from haolib.lib_nlp.vocabulary import Vocabulary
from haolib.lib_nlp.vectorizer import Vectorizer
from week1.utils.utility import *

prj_dir = get_cfd(backward=2)


def understanding_data():
    pd_helper = PandaHelper("{}/week1/data/validation.tsv".format(prj_dir), sep="\t")
    pd_helper.info()
    # test = read_data("{}/week1/data/validation.tsv".format(prj_dir))
    # print(test.head())


def prepare_data(base_path="./"):
    train = read_data('{}/train.tsv'.format(base_path))
    validation = read_data('{}/validation.tsv'.format(base_path))
    test = pd.read_csv('{}/test.tsv'.format(base_path), sep='\t')
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    return X_train, y_train, X_val, y_val, X_test


class BagOfWords(Vectorizer):
    def __init__(self, words_vocab, tags_vocab):
        super(BagOfWords, self).__init__(words_vocab, tags_vocab)

    def vectorize(self, text):
        """
                text: a string
                dict_size: size of the dictionary

                return a vector which is a bag-of-words representation of 'text'
            """
        result_vector = np.zeros(self.words_size)
        for word in text.split(' '):
            if word in self.words_vocab.token_to_idx:
                result_vector[self.words_vocab.token_to_idx[word]] += 1
        return result_vector

    def vectorize_multi(self, list_text):
        return [self.vectorize(text) for text in list_text]

class TfIdfVectorizer(Vectorizer):
    def __init__(self, words_vocab, tags_vocab):
        super(BagOfWords, self).__init__(words_vocab, tags_vocab)

    def vectorize(self, text):
        """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
        """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result


    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5, token_pattern='(\S+)') ####### YOUR CODE HERE #######

    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)


    def vectorize_multi(self, list_text):
        return [self.vectorize(text) for text in list_text]
if __name__ == "__main__":
    # understanding_data()
    X_train, y_train, X_val, y_val, X_test = prepare_data()
    print(X_train[:3])
    print(y_train[:3])
    most_common_words, most_common_tags = count_words_tags(X_train, y_train)
    print(most_common_words)
    print(most_common_tags)
