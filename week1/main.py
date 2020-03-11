# https://colab.research.google.com/drive/1jc0H_4lZelukMyFn_Y77lgmiN1DZ8doK
import nltk
from week1.utils.utility import *
# nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter

from week1.utils.dataset import prepare_data, BagOfWords
from haolib.lib_nlp.vocabulary import Vocabulary
from haolib.lib_nlp.vectorizer import Vectorizer
from haolib import *

if __name__ == "__main__":
    try:
        X_train_mybag, X_val_mybag, X_test_mybag = load_context("week1/data/data_train.pkl")
    except FileNotFoundError:
        X_train, y_train, X_val, y_val, X_test = prepare_data(base_path = "week1/data")
        words_counts = Counter([word for line in X_train for word in line.split(' ')])
        tags_counts = Counter([item for taglist in y_train for item in taglist])

        commons_words = words_counts.most_common(5000)
        commons_tags = tags_counts.most_common(200)
        word_to_idx = {item[0]: ii for ii, item in enumerate(sorted(commons_words, key=lambda x: x[1], reverse=True))}
        tag_to_idx = {item[0]: ii for ii, item in enumerate(sorted(commons_tags, key=lambda x: x[1], reverse=True))}

        vectorizer = BagOfWords(Vocabulary(token_to_idx=word_to_idx),
                                Vocabulary(token_to_idx=tag_to_idx))
        X_train_bags = vectorizer.vectorize_multi(X_train)
        X_val_bags = vectorizer.vectorize_multi(X_val)
        X_test_bags = vectorizer.vectorize_multi(X_test)

        X_train_mybag, X_val_mybag, X_test_mybag = convert_to_scipy(X_train_bags, X_val_bags, X_test_bags)
        save_context((X_train_mybag, X_val_mybag, X_test_mybag), file="week1/data/data_train.pkl")
    print(X_train_mybag[0:3])
    # print(vocabulary.lookup_index(1))
    # print(vocab)
