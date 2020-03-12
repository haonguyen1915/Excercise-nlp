from collections import defaultdict
import torch
from haolib import *
from torch.nn import functional as F


def read_data(file_path):
    tokens = []
    tags = []

    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            # Replace all urls with <URL> token
            # Replace all users with <USR> token

            ######################################
            ######### YOUR CODE HERE #############
            ######################################
            if token.startswith('@'):
                token = '<USR>'
            elif token.lower().startswith('http://') or token.lower().startswith('https://'):
                token = '<URL>'

            tweet_tokens.append(token)
            tweet_tags.append(tag)

    return tokens, tags


# def build_dict(tokens_or_tags, special_tokens):
#     """
#         tokens_or_tags: a list of lists of tokens or tags
#         special_tokens: some special tokens
#     """
#     # Create a dictionary with default value 0
#     tok2idx = {}
#     idx2tok = []
#
#     # Create mappings from tokens to indices and vice versa
#     # Add special tokens to dictionaries
#     # The first special token must have index 0
#
#     ######################################
#     ######### YOUR CODE HERE #############
#     ######################################
#     vocab = set([t for ts in tokens_or_tags for t in ts])
#     voab_size = len(vocab) + len(special_tokens)
#     idx2tok = [''] * voab_size
#
#     for i, token in enumerate(special_tokens):
#         tok2idx[token] = i
#         idx2tok[i] = token
#
#     for i, token in enumerate(vocab, len(special_tokens)):
#         print(i)
#         exit()
#         tok2idx[token] = i
#         idx2tok[i] = token
#
#     return tok2idx, idx2tok


def build_dict(tokens_or_tags):
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
    """
    # Create a dictionary with default value 0
    tok2idx = {}
    idx2tok = []

    vocab = set([t for ts in tokens_or_tags for t in ts])
    voab_size = len(vocab)
    idx2tok = [''] * voab_size

    # for i, token in enumerate(special_tokens):
    #     tok2idx[token] = i
    #     idx2tok[i] = token

    for i, token in enumerate(vocab):
        tok2idx[token] = i
        idx2tok[i] = token

    return tok2idx, idx2tok

def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes

    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def compute_accuracy(y_pred, y_true, mask_index=20503):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()
    acc = n_correct / n_valid * 100
    acc = torch.FloatTensor([acc])
    return acc


def sequence_loss(y_pred, y_true, mask_index=20503):

    y_pred, y_true = normalize_sizes(y_pred, y_true)
    # print(mask_index)
    # exit()
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)
