import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3 import PPO
'''
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

'''
class CustomCNNLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, sequence_length: int = 30, 
                 embed_dim: int = 128, num_heads: int = 4, num_layers_vit: int = 4, 
                 num_layers_temporal: int = 2, patch_size: int = 8):
        super(CustomCNNLSTMExtractor, self).__init__(observation_space, features_dim)
        
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers_vit = num_layers_vit
        self.num_layers_temporal = num_layers_temporal
        self.patch_size = patch_size
        
        n_input_channels = observation_space.shape[0]
        
        # Define CNN
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
        
        # Define LSTM with parameters
        self.lstm = nn.LSTM(n_flatten, self.embed_dim, num_layers=self.num_layers_temporal, batch_first=True)
        self.linear = nn.Linear(self.embed_dim, features_dim)
    
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

class EnhancedCNNLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, sequence_length: int = 30,
                 embed_dim: int = 256, num_heads: int = 8, num_layers_vit: int = 6,
                 num_layers_temporal: int = 2, patch_size: int = 8):
        super(EnhancedCNNLSTMExtractor, self).__init__(observation_space, features_dim)
        
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers_vit = num_layers_vit
        self.num_layers_temporal = num_layers_temporal
        self.patch_size = patch_size

        self.info = info
        
        
        n_input_channels = observation_space.shape[0]
        
        # Enhanced CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        # Compute the size of the flattened output
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # Define LSTM with parameters
        self.lstm = nn.LSTM(n_flatten, self.embed_dim, num_layers=self.num_layers_temporal, batch_first=True, dropout=0.3)

        # Add attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.1)
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Final linear layers
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.fc2 = nn.Linear(self.embed_dim // 2, features_dim)
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        #print(f"Input tensor shape: {observations.shape}")
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
        
        # Apply attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        attn_output = self.layer_norm(attn_output)

        # Add residual connection and layer normalization
        output = self.layer_norm(lstm_out + attn_output)

        

        # Final fully connected layers with ReLU activation
        output = F.relu(self.fc1(output[:, -1, :]))
        output = self.fc2(output)
        
        return output

class EnhancedCNNLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(EnhancedCNNLSTMPolicy, self).__init__(*args, **kwargs, features_extractor_class=EnhancedCNNLSTMExtractor)


        