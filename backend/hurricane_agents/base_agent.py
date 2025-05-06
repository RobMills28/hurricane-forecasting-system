"""
Base Agent for Hurricane Prediction

This module provides the base agent class that all specialized hurricane agents will extend.
"""

import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any, Union

# Define Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity=10000):
        """Initialize replay buffer with given capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample batch of experiences."""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Unzip experiences into separate arrays
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = [e.next_state for e in experiences]
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)

class QNetwork(nn.Module):
    """Neural network for Q-function approximation."""
    
    def __init__(self, state_dim, action_dim):
        """Initialize network with given dimensions."""
        super(QNetwork, self).__init__()
        
        # Deeper network architecture with dropout for regularization
        self.fc1 = nn.Linear(state_dim, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 256)  # Wider middle layer
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)  # Additional layer
        self.fc4 = nn.Linear(128, action_dim)
        
        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, x):
        """Forward pass through network."""
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class BaseAgent:
    """Base class for all hurricane prediction agents."""
    
    def __init__(self, options: Dict = None):
        """
        Initialize the base agent with configuration options.
        
        Args:
            options: Configuration options for the agent
        """
        # Default configuration options
        self.options = {
            "use_basin_models": True,       # Use basin-specific models
            "ensemble_size": 5,             # Number of sub-models in ensemble
            "learning_rate": 0.001,         # Learning rate for optimizer
            "gamma": 0.99,                  # Discount factor
            "epsilon_start": 1.0,           # Initial exploration rate
            "epsilon_end": 0.05,            # Final exploration rate
            "epsilon_decay": 0.995,         # Decay rate for exploration
            "buffer_size": 10000,           # Replay buffer size
            "batch_size": 64,               # Batch size for training
            "update_frequency": 4,          # How often to update the network
            "target_update_frequency": 1000, # How often to update target network
            "state_dim": 12,                # State dimension
            "action_dim": 15                # Action dimension
        }
        
        # Update with provided options
        if options:
            self.options.update(options)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize exploration rate
        self.epsilon = self.options["epsilon_start"]
        
        # Initialize step counter
        self.steps = 0
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.options["buffer_size"])
        
        # Initialize Q-networks (online and target)
        self.q_network = QNetwork(self.options["state_dim"], self.options["action_dim"]).to(self.device)
        self.target_network = QNetwork(self.options["state_dim"], self.options["action_dim"]).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.options["learning_rate"])
        
        # Basin-specific models
        self.basin_models = {
            "NA": self._create_basin_model(),  # North Atlantic
            "EP": self._create_basin_model(),  # Eastern Pacific
            "WP": self._create_basin_model(),  # Western Pacific
            "NI": self._create_basin_model(),  # North Indian
            "SI": self._create_basin_model(),  # South Indian
            "SP": self._create_basin_model(),  # South Pacific
            "DEFAULT": self._create_basin_model()  # Default model
        }
        
        # Initialize basin-specific movement patterns
        self._initialize_basin_patterns()
        
        # Performance tracking
        self.training_performance = []
        
        # Ensemble models
        self.ensemble_members = self._initialize_ensemble()
    
    def _initialize_basin_patterns(self):
        """Initialize basin-specific movement patterns."""
        self.basin_patterns = {
            "NA": {  # North Atlantic
                "early_lat_change": 0.15,  # Movement northward early
                "early_lon_change": -0.45,  # Strong westward component early
                "recurve_latitude": 25.0,   # Typical recurvature latitude
                "recurve_strength": 0.35,   # How sharply storms recurve
                "dissipation_latitude": 40.0,  # Where storms typically dissipate
                "intensity_change_rate": 0.08   # Intensification rate
            },
            "EP": {  # Eastern Pacific
                "early_lat_change": 0.1,
                "early_lon_change": -0.5,
                "recurve_latitude": 20.0,
                "recurve_strength": 0.25,
                "dissipation_latitude": 35.0,
                "intensity_change_rate": 0.06
            },
            "WP": {  # Western Pacific
                "early_lat_change": 0.1,
                "early_lon_change": -0.3,
                "recurve_latitude": 22.0,
                "recurve_strength": 0.4,
                "dissipation_latitude": 45.0,
                "intensity_change_rate": 0.1  # Greater intensification in WP
            },
            "DEFAULT": {
                "early_lat_change": 0.1,
                "early_lon_change": -0.4,
                "recurve_latitude": 25.0,
                "recurve_strength": 0.3,
                "dissipation_latitude": 40.0,
                "intensity_change_rate": 0.08
            }
        }
        
    def _create_basin_model(self):
        """Create a basin-specific model."""
        q_network = QNetwork(self.options["state_dim"], self.options["action_dim"]).to(self.device)
        target_network = QNetwork(self.options["state_dim"], self.options["action_dim"]).to(self.device)
        target_network.load_state_dict(q_network.state_dict())
        target_network.eval()
        
        optimizer = optim.Adam(q_network.parameters(), lr=self.options["learning_rate"])
        
        return {
            "q_network": q_network,
            "target_network": target_network,
            "optimizer": optimizer,
            "steps": 0
        }
    
    def _initialize_ensemble(self):
        """Initialize ensemble models with slight variations."""
        ensemble = []
        
        for i in range(self.options["ensemble_size"]):
            # Create variation of learning rate and architecture
            variation_factor = 0.1 + (random.random() * 0.1)
            lr = self.options["learning_rate"] * (1 + (random.random() * variation_factor * 2 - variation_factor))
            
            # Create ensemble member
            q_network = QNetwork(self.options["state_dim"], self.options["action_dim"]).to(self.device)
            target_network = QNetwork(self.options["state_dim"], self.options["action_dim"]).to(self.device)
            target_network.load_state_dict(q_network.state_dict())
            target_network.eval()
            
            optimizer = optim.Adam(q_network.parameters(), lr=lr)
            
            ensemble.append({
                "q_network": q_network,
                "target_network": target_network,
                "optimizer": optimizer,
                "steps": 0,
                "epsilon": self.options["epsilon_start"] * (1 + (random.random() * 0.2 - 0.1)),
                "buffer": ReplayBuffer(self.options["buffer_size"] // 2)  # Smaller buffer
            })
        
        return ensemble
    
    def preprocess_state(self, state: Dict) -> torch.Tensor:
        """Convert state dictionary to tensor representation."""
        # Extract features from state
        position = state.get("position", {})
        lat = position.get("lat", 0)
        lon = position.get("lon", 0)
        wind_speed = state.get("wind_speed", 0)
        pressure = state.get("pressure", 1010)
        
        # Get sea surface temperature
        sst = state.get("sea_surface_temp", {}).get("value", 28)
        
        # Time step (normalize)
        time_step = state.get("time_step", 0) / 20  # Normalize
        
        # Basin one-hot encoding (6 basins)
        basin = state.get("basin", "DEFAULT")
        basin_encoding = [0] * 6
        basin_idx = {"NA": 0, "EP": 1, "WP": 2, "NI": 3, "SI": 4, "SP": 5}.get(basin, 0)
        basin_encoding[basin_idx] = 1
        
        # Normalize features
        lat_norm = lat / 90.0  # Normalize latitude
        lon_norm = lon / 180.0  # Normalize longitude
        wind_speed_norm = wind_speed / 200.0  # Normalize wind speed
        pressure_norm = (pressure - 880) / (1020 - 880)  # Normalize pressure
        sst_norm = (sst - 20) / (32 - 20)  # Normalize SST
        
        # Create state vector
        state_vector = [
            lat_norm, lon_norm, wind_speed_norm, pressure_norm, sst_norm, time_step,
            *basin_encoding
        ]
        
        # Convert to tensor
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def predict(self, state: Dict, history: List[Dict], training: bool = False) -> Dict:
        """
        Make a prediction based on current state and history.
        
        Args:
            state: Current hurricane state
            history: List of historical states
            training: Whether prediction is for training
            
        Returns:
            Prediction dictionary with position, wind speed, and pressure
        """
        # This method should be overridden by specialized agents
        raise NotImplementedError("Predict method must be implemented by specialized agents")
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's Q-network based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add experience to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Only update if we have enough samples
        if len(self.replay_buffer) < self.options["batch_size"]:
            return
        
        # Only update every update_frequency steps
        if self.steps % self.options["update_frequency"] != 0:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.options["batch_size"])
        
        # Preprocess states
        state_tensors = torch.cat([self.preprocess_state(s) for s in states])
        action_tensors = torch.tensor([self._prediction_to_action(a) for a in actions], 
                                     dtype=torch.long).unsqueeze(1).to(self.device)
        next_state_tensors = torch.cat([self.preprocess_state(s) for s in next_states])
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # Get current Q values
        current_q_values = self.q_network(state_tensors).gather(1, action_tensors)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensors).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (self.options["gamma"] * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps % self.options["target_update_frequency"] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _action_to_prediction(self, action_idx, state):
        """Convert discrete action index to continuous prediction."""
        # This method would be overridden by specialized agents
        raise NotImplementedError("_action_to_prediction must be implemented by specialized agents")
    
    def _prediction_to_action(self, prediction):
        """Convert continuous prediction back to discrete action index."""
        # This method would be overridden by specialized agents
        raise NotImplementedError("_prediction_to_action must be implemented by specialized agents")
    
    def calculate_sst_effect(self, sst: float) -> float:
        """Calculate the effect of sea surface temperature on intensification."""
        # Enhanced SST thresholds based on research
        if sst < 25:
            return -0.15  # Stronger weakening effect
        if sst < 26:
            return -0.08  # Moderate weakening
        if sst < 27:
            return 0.0    # Neutral effect
        if sst < 28:
            return 0.05   # Slight intensification 
        if sst < 29:
            return 0.12   # Moderate intensification
        if sst < 30:
            return 0.20   # Strong intensification
        return 0.30       # Very strong intensification
    
    def calculate_latitude_effect(self, latitude: float, basin: str) -> float:
        """Calculate the effect of latitude on hurricane behavior."""
        # Normalize latitude effect based on basin
        abs_lat = abs(latitude)
        
        lat_threshold = 30  # Default
        if basin == "WP":  # Western Pacific typhoons can sustain at higher latitudes
            lat_threshold = 35
        elif basin == "NA":  # North Atlantic
            lat_threshold = 30
        else:
            lat_threshold = 28
        
        # No effect near equator
        if abs_lat < 15:
            return 0
        
        # Increasing effect as latitude increases
        if abs_lat < lat_threshold:
            return (abs_lat - 15) / (lat_threshold - 15) * 0.5
        
        # Strong effect beyond threshold
        return 0.5 + ((abs_lat - lat_threshold) / 10) * 0.5
    
    def _filter_outliers(self, values: List[float]) -> List[float]:
        """Remove outliers from a list of values (beyond 2 standard deviations)."""
        if not values or len(values) < 3:
            return values
            
        mean = sum(values) / len(values)
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
        threshold = 2 * std_dev
        
        return [x for x in values if abs(x - mean) <= threshold]
    
    def _calculate_std_dev(self, values: List[float], mean: float) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
            
        square_diffs = [(value - mean) ** 2 for value in values]
        avg_square_diff = sum(square_diffs) / len(values)
        return math.sqrt(avg_square_diff)