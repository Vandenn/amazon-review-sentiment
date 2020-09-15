from torch import nn

class Model1(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim=256, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded_text = self.embedding(text)
        output, hidden = self.rnn(embedded_text)
        return self.fc(hidden.squeeze(0))
