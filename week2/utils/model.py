import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import tensor
from haolib import *
_logger = LogTracker()
_logger.disable_print_log()


class LSTMNERClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMNERClassifier, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


if __name__ == "__main__":
    model = LSTMNERClassifier(embedding_dim=32,
                              hidden_dim=32,
                              vocab_size=88,
                              target_size=10)
    input = torch.ones().view(10)
    output = model(input)
    describe_tensor(output)

