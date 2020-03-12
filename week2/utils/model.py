import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import tensor
from haolib import *
_logger = LogTracker()
_logger.disable_print_log()
class NERClassifierModel(nn.Module):
    def __init__(self, char_embedding_size, char_vocab_size, out_feature_size, rnn_hidden_size,
                 batch_first=True, padding_idx=0, dropout_p=0.5):
        """
        Args:
            char_embedding_size (int): The size of the character embeddings
            char_vocab_size (int): The number of characters to embed
            rnn_hidden_size (int): The size of the RNN's hidden state
            batch_first (bool): Informs whether the input tensors will
                have batch or the sequence on the 0th dimension
            padding_idx (int): The index for the tensor padding;
                see torch.nn.Embedding
            dropout_p (float): the probability of zeroing activations using
                the dropout method.  higher means more likely to zero.
        """
        super(NERClassifierModel, self).__init__()

        self.char_emb = nn.Embedding(num_embeddings=char_vocab_size,
                                     embedding_dim=char_embedding_size,
                                     padding_idx=padding_idx)

        self.rnn = nn.GRU(input_size=char_embedding_size,
                          hidden_size=rnn_hidden_size,
                          batch_first=batch_first)

        self.fc = nn.Linear(in_features=rnn_hidden_size,
                            out_features=out_feature_size)

        self._dropout_p = dropout_p

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the model

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, char_vocab_size)
        """
        x_embedded = self.char_emb(x_in)
        y_out, _ = self.rnn(x_embedded)
        batch_size, seq_size, feat_size = y_out.shape
        y_out = y_out.contiguous().view(batch_size * seq_size, feat_size)
        y_out = self.fc(F.dropout(y_out, p=self._dropout_p))
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)

        new_feat_size = y_out.shape[-1]
        y_out = y_out.view(batch_size, seq_size, new_feat_size)
        return y_out


if __name__ == "__main__":
    model = NERClassifierModel(char_embedding_size=32,
                                   char_vocab_size=88,
                                   out_feature_size=10,
                                   rnn_hidden_size=32,
                                   padding_idx=0)
    input = tensor([[2, 49, 18, 15, 18, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 13, 7, 14, 8, 5, 11, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 24, 5, 33, 11, 21, 51, 31, 23, 14, 5, 53, 7, 0, 0, 0, 0, 0, 0],
                    [2, 34, 7, 14, 7, 6, 5, 25, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    output = model(input)
    describe_tensor(output)

