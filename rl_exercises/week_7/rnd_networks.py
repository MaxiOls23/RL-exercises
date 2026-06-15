import torch.nn as nn


class RNDNetwork(nn.Module):
    def __init__(self, input_shape, output_size, hidden_size=128):
        super(RNDNetwork, self).__init__()

        self.target_network = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.predictor_network = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # Initialize target network with fixed random weights
        for param in self.target_network.parameters():
            param.requires_grad = False

    def forward(self, x):
        target_output = self.target_network(x)
        predictor_output = self.predictor_network(x)
        return target_output, predictor_output

    def predict(self, x):
        return self.predictor_network(x)

    def target(self, x):
        return self.target_network(x)
