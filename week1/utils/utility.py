import pandas as pd
from ast import literal_eval
from scipy import sparse as sp_sparse
from week1.utils.constants import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])  # delete stopwors from text
    text = text.strip()

    return text


def convert_to_scipy(X_train, X_val, X_test):
    X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(text) for text in X_train])
    X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(text) for text in X_val])
    X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(text) for text in X_test])
    print('X_train shape ', X_train_mybag.shape)
    print('X_val shape ', X_val_mybag.shape)
    print('X_test shape ', X_test_mybag.shape)
    return X_train_mybag, X_val_mybag, X_test_mybag


def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    ######################################
    ######### YOUR CODE HERE #############
    ######################################


    model = OneVsRestClassifier(LogisticRegression(penalty='l2', C=1.0))
    model.fit(X_train, y_train)
    return model
