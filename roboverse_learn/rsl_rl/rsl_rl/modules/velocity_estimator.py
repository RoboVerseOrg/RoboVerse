import torch.nn as nn


class VelocityEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, history_len):
        super(VelocityEstimator, self).__init__()
        self.fc1 = nn.Linear(history_len * input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch.nn as nn

# class VelocityEstimatorGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(VelocityEstimatorGRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#         # self.h0 = torch.zeros(1, 25, self.hidden_size, requires_grad=True).to('cuda:1')
#         self.relu = nn.ReLU()

#     def forward(self, x):
#     # x should have shape (batch_size, sequence_length, input_size)
#         batch_size = x.size(0)
#         sequence_length = x.size(1)

#         # Initialize the initial hidden state with zeros
#         h0 = torch.zeros(2, sequence_length, self.hidden_size).to(x.device)

#         # Pass the input tensor through the GRU layer
#         out, _ = self.gru(x, h0)

#         # Take the output from the last time step
#         out = self.relu(out[:, -1, :])

#         # Pass the output through the fully connected layer
#         out = self.fc(out)

#         return out


class VelocityEstimatorGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VelocityEstimatorGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.hidden_states = None

    def forward(self, x, batch_mode=True, hidden_states=None):
        if batch_mode:
            # Pass the input tensor through the GRU layer
            out, _ = self.gru(x, hidden_states)

            # Take the output from the last time step
            out = self.relu(out[:, -1, :])

            # Pass the output through the fully connected layer
            out = self.fc(out)
        else:
            # Pass the input tensor through the GRU layer
            out, self.hidden_states = self.gru(x.unsqueeze(0), self.hidden_states)

            # Take the output from the last time step
            out = self.relu(out[:, -1, :])

            # Pass the output through the fully connected layer
            out = self.fc(out)

        return out

    def reset(self, dones=None):
        self.hidden_states = None
