import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomCNNLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNNLSTMExtractor, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.lstm = nn.LSTM(n_flatten, 256, batch_first=True)

        self.linear = nn.Linear(256, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]

        cnn_output = self.cnn(observations)
        cnn_output = cnn_output.view(batch_size, 1, -1)

        lstm_out, _ = self.lstm(cnn_output)

        output = self.linear(lstm_out[:, -1, :])

        return output

class CustomCNNLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCNNLSTMPolicy, self).__init__(*args, **kwargs, features_extractor_class=CustomCNNLSTMExtractor)