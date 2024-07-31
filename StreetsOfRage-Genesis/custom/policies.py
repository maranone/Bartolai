import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy



class CustomCNNLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, sequence_length: int = 15):
        super(CustomCNNLSTMExtractor, self).__init__(observation_space, features_dim)
        
        self.sequence_length = sequence_length
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute the size of the flattened output
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.lstm = nn.LSTM(n_flatten, 256, batch_first=True)
        self.linear = nn.Linear(256, features_dim)
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        
        if observations.dim() == 4:
            batch_size, C, H, W = observations.shape
            sequence_length = 1
            observations = observations.unsqueeze(1)  # Add sequence dimension
        else:
            batch_size, sequence_length, C, H, W = observations.shape


        # Process each frame through CNN
        cnn_outputs = []
        for t in range(sequence_length):
            frame_output = self.cnn(observations[:, t])  # Process t-th frame
            cnn_outputs.append(frame_output)
        
        cnn_output = th.stack(cnn_outputs, dim=1)  # Shape: (batch_size, sequence_length, feature_dim)
        
        lstm_out, _ = self.lstm(cnn_output)  # Process through LSTM
        output = self.linear(lstm_out[:, -1, :])  # Use the output of the last time step

        return output

class CustomCNNLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCNNLSTMPolicy, self).__init__(*args, **kwargs, features_extractor_class=CustomCNNLSTMExtractor)
