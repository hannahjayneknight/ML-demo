import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forwardOLD(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.rnn(x, h0.reshape(self.num_layers, x.size(0), self.hidden_size))
        out = self.fc(out[:, -1, :])

        return out

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.rnn(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])

        return out
