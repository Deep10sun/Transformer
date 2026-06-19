import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(embedding_size,embedding_size)

    def forward(self, indices, target=None):

        logits = self.embedding(indices)

        return logits