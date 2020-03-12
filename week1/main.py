# https://colab.research.google.com/drive/1jc0H_4lZelukMyFn_Y77lgmiN1DZ8doK
# https://colab.research.google.com/drive/1KQ85Y_yOI30vUqsece7SdInA4JRrvZzS#scrollTo=RJgl_gKPuzRo
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
    X_train, y_train, X_val, y_val, X_test = prepare_data(base_path="week1/data")
    tags_counts = Counter([item for taglist in y_train for item in taglist])

    try:
        X_train_mybag, X_val_mybag, X_test_mybag = load_context("week1/data/data_train.pkl")
    except FileNotFoundError:
        words_counts = Counter([word for line in X_train for word in line.split(' ')])

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

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)

    classifier_mybag = train_classifier(X_train_mybag, y_train)
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    # y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    # y_val_inversed = mlb.inverse_transform(y_val)
    # for i in range(3):
    #     print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
    #         X_val[i],
    #         ','.join(y_val_inversed[i]),
    #         ','.join(y_val_pred_inversed[i])
    #     ))
    #
    # print(vocabulary.lookup_index(1))
    # print(vocab)
