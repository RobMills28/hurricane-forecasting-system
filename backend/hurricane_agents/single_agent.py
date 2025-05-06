"""
Hurricane Prediction Agent with Reinforcement Learning

This module provides agent implementations for hurricane trajectory and intensity prediction
using Deep Q-Learning and ensemble methods.
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
    """Enhanced neural network for Q-function approximation."""
    
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

class HurricanePredictionAgent:
    """
    Hurricane Prediction Agent using Deep Q-Learning with ensemble methods
    and basin-specific models.
    """
    
    def __init__(self, options: Dict = None):
        """
        Initialize the hurricane prediction agent.
        
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
            "action_dim": 15                # Action dimension (5 lat dirs, 3 lon dirs, 1 intensity)
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
        
        # Environmental factor weights
        self.environmental_factors = {
            "sea_surface_temp": 0.6,         # Increased influence of SST on intensification
            "latitudinal_effect": 0.35,      # Increased effect of latitude on track
            "seasonal_effect": 0.15,         # Slightly increased seasonal influence
            "basin_climatology": 0.25        # New factor for basin-specific climatology
            
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
                "buffer": ReplayBuffer(self.options["buffer_size"] // 2)  # Smaller buffer for ensemble members
            })
        
        return ensemble
    
    def preprocess_state(self, state: Dict) -> torch.Tensor:
        """Convert state dictionary to tensor representation suitable for neural network."""
        # Extract features from state
        position = state.get("position", {})
        lat = position.get("lat", 0)
        lon = position.get("lon", 0)
        wind_speed = state.get("wind_speed", 0)
        pressure = state.get("pressure", 1010)
        
        # Get sea surface temperature
        sst = state.get("sea_surface_temp", {}).get("value", 28)
        
        # Time step (normalize)
        time_step = state.get("time_step", 0) / 20  # Normalize assuming max of 20 time steps
        
        # Basin one-hot encoding (6 basins)
        basin = state.get("basin", "DEFAULT")
        basin_encoding = [0] * 6
        basin_idx = {"NA": 0, "EP": 1, "WP": 2, "NI": 3, "SI": 4, "SP": 5}.get(basin, 0)
        basin_encoding[basin_idx] = 1
        
        # Normalize features
        lat_norm = lat / 90.0  # Normalize latitude
        lon_norm = lon / 180.0  # Normalize longitude
        wind_speed_norm = wind_speed / 200.0  # Normalize wind speed (0-200 mph)
        pressure_norm = (pressure - 880) / (1020 - 880)  # Normalize pressure (880-1020 mb)
        sst_norm = (sst - 20) / (32 - 20)  # Normalize SST (20-32°C)
        
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
        # Increment step counter
        self.steps += 1
        
        # Get basin-specific model if available
        basin = state.get("basin", "DEFAULT")
        model = self.basin_models.get(basin) if self.options["use_basin_models"] else None
        
        # If model is None, fall back to default model
        if model is None:
            model = self.basin_models.get("DEFAULT")
        
        # Decay epsilon
        if training:
            self.epsilon = max(
                self.options["epsilon_end"],
                self.epsilon * self.options["epsilon_decay"]
            )
        
        # Preprocess state
        state_tensor = self.preprocess_state(state)
        
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            action_idx = random.randint(0, self.options["action_dim"] - 1)
        else:
            # Use Q-network for prediction
            with torch.no_grad():
                if model:
                    q_values = model["q_network"](state_tensor)
                else:
                    q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        # Convert action index to prediction
        prediction = self._action_to_prediction(action_idx, state)
        
        # Make individual ensemble predictions if not training
        if not training:
            ensemble_predictions = [
                self._ensemble_member_predict(member, state)
                for member in self.ensemble_members
            ]
            
            # Add main model prediction
            ensemble_predictions.append(prediction)
            
            # Combine ensemble predictions
            return self.combine_ensemble_predictions(ensemble_predictions, state)
        
        return prediction
    
    def _ensemble_member_predict(self, member, state):
        """Make prediction using ensemble member."""
        # Preprocess state
        state_tensor = self.preprocess_state(state)
        
        # Use member's Q-network
        with torch.no_grad():
            q_values = member["q_network"](state_tensor)
            action_idx = q_values.argmax().item()
        
        # Convert action index to prediction
        return self._action_to_prediction(action_idx, state)
    
    def _action_to_prediction(self, action_idx, state):
        """Convert discrete action index to continuous prediction."""
        # Extract current position and intensity
        position = state.get("position", {})
        current_lat = position.get("lat", 0)
        current_lon = position.get("lon", 0)
        current_wind = state.get("wind_speed", 0)
        current_pressure = state.get("pressure", 1010)
        
        # Decode action
        # We have 15 actions:
        # - 5 latitude changes (---, --, -, +, ++)
        # - 3 longitude changes (-, 0, +)
        # - 1 intensity change (-, 0, +)
        
        # Calculate lat/lon changes based on action
        lat_action = action_idx // 9  # 0-4
        lon_action = (action_idx % 9) // 3  # 0-2
        intensity_action = action_idx % 3  # 0-2
        
        # Convert actions to changes
        lat_changes = [-1.0, -0.5, 0.0, 0.5, 1.0]
        lon_changes = [-0.5, 0.0, 0.5]
        intensity_changes = [-10.0, 0.0, 10.0]
        
        lat_change = lat_changes[lat_action]
        lon_change = lon_changes[lon_action]
        intensity_change = intensity_changes[intensity_action]
        
        # Get basin patterns
        basin = state.get("basin", "DEFAULT")
        patterns = self.basin_patterns.get(basin, self.basin_patterns["DEFAULT"])

        # Modify changes based on basin patterns
        if current_lat < patterns["recurve_latitude"]:
            # Early track behavior (before recurvature)
            lat_change = lat_change * 0.7 + patterns["early_lat_change"] * 0.3
            lon_change = lon_change * 0.7 + patterns["early_lon_change"] * 0.3
        else:
            # Apply recurvature effect
            recurve_effect = (current_lat - patterns["recurve_latitude"]) * patterns["recurve_strength"] / 10.0
            lat_change = lat_change + recurve_effect * 0.4
            lon_change = lon_change + recurve_effect * 0.6  # Stronger eastward component after recurvature

        # Modify intensity change based on basin patterns
        intensity_change = intensity_change * (1 + patterns["intensity_change_rate"])

        # Apply modified changes
        new_lat = current_lat + lat_change
        new_lon = current_lon + lon_change
        new_wind = max(0, current_wind + intensity_change)
        
        # Ensure latitude is within bounds
        new_lat = max(-90, min(90, new_lat))
        
        # Ensure longitude is within bounds
        new_lon = ((new_lon + 180) % 360) - 180
        
        # Calculate new pressure based on wind-pressure relationship
        new_pressure = self._calculate_pressure_from_wind(new_wind)
        
        # Apply environmental factors
        prediction = self._apply_environmental_factors({
            "position": {"lat": new_lat, "lon": new_lon},
            "wind_speed": new_wind,
            "pressure": new_pressure
        }, state)
        
        return prediction
    
    def _calculate_pressure_from_wind(self, wind_speed):
        """Calculate pressure from wind speed using wind-pressure relationship."""
        # P = 1010 - (wind_speed / 1.15)^2 / 100
        return max(880, min(1020, 1010 - (wind_speed / 1.15)**2 / 100))
    
    def _apply_environmental_factors(self, prediction, state):
        """Apply environmental factors to prediction."""
        # Extract prediction values
        position = prediction.get("position", {})
        lat = position.get("lat", 0)
        lon = position.get("lon", 0)
        wind_speed = prediction.get("wind_speed", 0)
        pressure = prediction.get("pressure", 1010)
        
        # Apply sea surface temperature effect
        sea_surface_temp = state.get("sea_surface_temp", {})
        if sea_surface_temp and "value" in sea_surface_temp:
            # SST effect on intensity
            sst_effect = self.calculate_sst_effect(sea_surface_temp["value"])
            wind_speed *= (1 + (sst_effect * self.environmental_factors["sea_surface_temp"]))
            
            # Higher SST typically means lower pressure
            pressure *= (1 - (sst_effect * self.environmental_factors["sea_surface_temp"] * 0.05))
        
        # Apply latitude effect (higher latitudes typically mean weakening and recurvature)
        latitude_effect = self.calculate_latitude_effect(lat, state.get("basin"))
        
        # Apply latitude effect to track and intensity
        if abs(lat) > 25:
            # More poleward movement at higher latitudes
            poleward_direction = 1 if lat > 0 else -1
            lat += latitude_effect * poleward_direction * self.environmental_factors["latitudinal_effect"]
            
            # Generally eastward movement at higher latitudes
            lon += latitude_effect * 0.2 * self.environmental_factors["latitudinal_effect"]
            
            # Weakening at higher latitudes
            wind_speed *= (1 - (latitude_effect * self.environmental_factors["latitudinal_effect"] * 0.1))
            pressure *= (1 + (latitude_effect * self.environmental_factors["latitudinal_effect"] * 0.02))
        
        # Limit pressure to realistic values
        pressure = max(880, min(1020, pressure))
        
        # Ensure wind speed and pressure are physically consistent
        # Lower pressure should correlate with higher wind speed
        pressure_wind_correlation = self._correlate_wind_and_pressure(wind_speed, pressure)
        
        return {
            "position": {"lat": lat, "lon": lon},
            "wind_speed": pressure_wind_correlation["wind_speed"],
            "pressure": pressure_wind_correlation["pressure"]
        }
    
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
        
        # Update each ensemble member on a subset of experiences
        if len(self.replay_buffer) >= self.options["batch_size"] // 2:
            for member in self.ensemble_members:
                self._update_ensemble_member(member)
    
    def _update_ensemble_member(self, member):
        """Update an ensemble member using its own buffer and experiences."""
        # Increment step counter
        member["steps"] += 1
        
        # Only update every update_frequency steps
        if member["steps"] % self.options["update_frequency"] != 0:
            return
        
        # Sample batch from member's buffer
        if len(member["buffer"]) < self.options["batch_size"] // 2:
            # If member's buffer is too small, use main buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.options["batch_size"] // 2)
        else:
            states, actions, rewards, next_states, dones = member["buffer"].sample(self.options["batch_size"] // 2)
        
        # Preprocess states
        state_tensors = torch.cat([self.preprocess_state(s) for s in states])
        action_tensors = torch.tensor([self._prediction_to_action(a) for a in actions], 
                                     dtype=torch.long).unsqueeze(1).to(self.device)
        next_state_tensors = torch.cat([self.preprocess_state(s) for s in next_states])
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # Get current Q values from member's Q-network
        current_q_values = member["q_network"](state_tensors).gather(1, action_tensors)
        
        # Get next Q values from member's target network
        with torch.no_grad():
            next_q_values = member["target_network"](next_state_tensors).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (self.options["gamma"] * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        member["optimizer"].zero_grad()
        loss.backward()
        member["optimizer"].step()
        
        # Update target network periodically
        if member["steps"] % self.options["target_update_frequency"] == 0:
            member["target_network"].load_state_dict(member["q_network"].state_dict())
    
    def _prediction_to_action(self, prediction):
        """Convert continuous prediction back to discrete action index."""
        # This is an approximate inverse of _action_to_prediction
        # Extract prediction values
        position = prediction.get("position", {})
        lat_change = position.get("lat", 0) - position.get("lat_prev", 0)
        lon_change = position.get("lon", 0) - position.get("lon_prev", 0)
        intensity_change = prediction.get("wind_speed", 0) - prediction.get("wind_speed_prev", 0)
        
        # Discretize lat change
        if lat_change <= -0.75:
            lat_action = 0  # ---
        elif lat_change <= -0.25:
            lat_action = 1  # --
        elif lat_change <= 0.25:
            lat_action = 2  # -
        elif lat_change <= 0.75:
            lat_action = 3  # +
        else:
            lat_action = 4  # ++
        
        # Discretize lon change
        if lon_change <= -0.25:
            lon_action = 0  # -
        elif lon_change <= 0.25:
            lon_action = 1  # 0
        else:
            lon_action = 2  # +
        
        # Discretize intensity change
        if intensity_change <= -5:
            intensity_action = 0  # -
        elif intensity_change <= 5:
            intensity_action = 1  # 0
        else:
            intensity_action = 2  # +
        
        # Calculate action index
        return lat_action * 9 + lon_action * 3 + intensity_action
    
    def combine_ensemble_predictions(self, predictions: List[Dict], state: Dict) -> Dict:
        """
        Combine predictions from ensemble members with improved weighting and outlier handling.
        
        Args:
            predictions: List of individual predictions
            state: Current state
            
        Returns:
            Combined prediction with uncertainty estimates
        """
        # Extract main model prediction (last element) and ensemble predictions
        main_prediction = predictions[-1] if predictions else {}
        ensemble_predictions = predictions[:-1] if len(predictions) > 1 else []
        
        if not ensemble_predictions:
            return main_prediction
        
        # Extract position values
        positions_lat = [pred.get("position", {}).get("lat", 0) for pred in ensemble_predictions]
        positions_lon = [pred.get("position", {}).get("lon", 0) for pred in ensemble_predictions]
        wind_speeds = [pred.get("wind_speed", 0) for pred in ensemble_predictions]
        pressures = [pred.get("pressure", 0) for pred in ensemble_predictions]
        
        # Get main model values
        main_lat = main_prediction.get("position", {}).get("lat", 0)
        main_lon = main_prediction.get("position", {}).get("lon", 0)
        main_wind = main_prediction.get("wind_speed", 0)
        main_pressure = main_prediction.get("pressure", 0)
        
        # Add main model to ensemble values with higher weight
        main_weight = 2.0  # Main model has 2x the weight of ensemble members
        
        # Add weighted main model values
        weighted_positions_lat = positions_lat + [main_lat] * int(main_weight)
        weighted_positions_lon = positions_lon + [main_lon] * int(main_weight)
        weighted_wind_speeds = wind_speeds + [main_wind] * int(main_weight)
        weighted_pressures = pressures + [main_pressure] * int(main_weight)
        
        # Remove outliers (values beyond 2 standard deviations)
        filtered_lat = self._filter_outliers(weighted_positions_lat)
        filtered_lon = self._filter_outliers(weighted_positions_lon)
        filtered_wind = self._filter_outliers(weighted_wind_speeds)
        filtered_pressure = self._filter_outliers(weighted_pressures)
        
        # Calculate means with outliers removed
        avg_lat = sum(filtered_lat) / len(filtered_lat) if filtered_lat else main_lat
        avg_lon = sum(filtered_lon) / len(filtered_lon) if filtered_lon else main_lon
        avg_wind = sum(filtered_wind) / len(filtered_wind) if filtered_wind else main_wind
        avg_pressure = sum(filtered_pressure) / len(filtered_pressure) if filtered_pressure else main_pressure
        
        # Calculate standard deviations for uncertainty estimation
        lat_std_dev = self._calculate_std_dev(filtered_lat, avg_lat)
        lon_std_dev = self._calculate_std_dev(filtered_lon, avg_lon)
        wind_std_dev = self._calculate_std_dev(filtered_wind, avg_wind)
        pressure_std_dev = self._calculate_std_dev(filtered_pressure, avg_pressure)
        
        return {
            "position": {"lat": avg_lat, "lon": avg_lon},
            "wind_speed": avg_wind,
            "pressure": avg_pressure,
            "uncertainty": {
                "position": {"lat": lat_std_dev, "lon": lon_std_dev},
                "wind_speed": wind_std_dev,
                "pressure": pressure_std_dev
            }
        }

    def _filter_outliers(self, values: List[float]) -> List[float]:
        """Remove outliers from a list of values (beyond 2 standard deviations)."""
        if not values or len(values) < 3:
            return values
            
        mean = sum(values) / len(values)
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
        threshold = 2 * std_dev
        
        return [x for x in values if abs(x - mean) <= threshold]
    
    def _calculate_std_dev(self, values: List[float], mean: float) -> float:
        """
        Calculate standard deviation.
        
        Args:
            values: List of values
            mean: Mean value
            
        Returns:
            Standard deviation
        """
        if not values:
            return 0.0
            
        square_diffs = [(value - mean) ** 2 for value in values]
        avg_square_diff = sum(square_diffs) / len(values)
        return math.sqrt(avg_square_diff)
    
    def calculate_sst_effect(self, sst: float) -> float:
        """
        Calculate the effect of sea surface temperature on hurricane intensification.
        
        Args:
            sst: Sea surface temperature in degrees Celsius
            
        Returns:
            Effect multiplier
        """
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
        return 0.30       # Very strong intensification (rapid intensification potential)
    
    def calculate_latitude_effect(self, latitude: float, basin: str) -> float:
        """
        Calculate the effect of latitude on hurricane behavior.
        
        Args:
            latitude: Latitude in degrees
            basin: Ocean basin identifier
            
        Returns:
            Latitude effect multiplier
        """
        # Normalize latitude effect based on basin
        # Different basins have different latitude thresholds
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
    
    def _correlate_wind_and_pressure(self, wind_speed: float, pressure: float) -> Dict:
        """
        Ensure wind speed and pressure are physically consistent.
        
        Args:
            wind_speed: Wind speed in mph
            pressure: Pressure in millibars
            
        Returns:
            Dictionary with adjusted wind speed and pressure
        """
        # Use a simplified version of the wind-pressure relationship
        # P = 1010 - (windSpeed/1.15)^2/100
        # where windSpeed is in mph and pressure is in millibars
        
        # Calculate "ideal" pressure from wind
        ideal_pressure = 1010 - (wind_speed/1.15)**2/100
        
        # Calculate "ideal" wind from pressure
        delta_pressure = max(0, 1010 - pressure)  # Ensure non-negative
        ideal_wind = 1.15 * math.sqrt(delta_pressure * 100)
        
        # Average the current prediction with the "ideal" values
        return {
            "wind_speed": (wind_speed + ideal_wind) / 2,
            "pressure": (pressure + ideal_pressure) / 2
        }
    
    async def train(self, environment, episodes=100) -> List[Dict]:
        """
        Train the agent on historical data.
        
        Args:
            environment: The hurricane environment
            episodes: Number of episodes to train
            
        Returns:
            List of performance metrics for each episode
        """
        print(f"Training hurricane prediction agent on {episodes} episodes...")
        performance = []
        
        for episode in range(episodes):
            # Reset the environment and get initial state
            state = environment.reset()
            
            done = False
            episode_reward = 0
            
            # Store previous state for tracking changes
            prev_state = None
            
            # Episode loop
            while not done:
                # Get current state
                current_state = environment.get_state()
                
                # Add previous state info for action conversion
                if prev_state:
                    current_state["position"]["lat_prev"] = prev_state.get("position", {}).get("lat", 0)
                    current_state["position"]["lon_prev"] = prev_state.get("position", {}).get("lon", 0)
                    current_state["wind_speed_prev"] = prev_state.get("wind_speed", 0)
                    current_state["pressure_prev"] = prev_state.get("pressure", 0)
                else:
                    # First step, use current as previous
                    current_state["position"]["lat_prev"] = current_state.get("position", {}).get("lat", 0)
                    current_state["position"]["lon_prev"] = current_state.get("position", {}).get("lon", 0)
                    current_state["wind_speed_prev"] = current_state.get("wind_speed", 0)
                    current_state["pressure_prev"] = current_state.get("pressure", 0)
                
                # Make prediction (in training mode)
                action = self.predict(current_state, [], training=True)
                
                # Take action and observe result
                result = environment.step(action)
                next_state = result.get("state", {})
                reward = result.get("reward", 0)
                done = result.get("done", False)
                
                # Update agent using RL
                self.update(current_state, action, reward, next_state, done)
                
                # Accumulate reward
                episode_reward += reward
                
                # Store state for history
                prev_state = current_state
                
                # Make current state the next state
                current_state = next_state
            
            # Evaluate performance
            episode_performance = environment.evaluate_performance()
            episode_performance["total_reward"] = episode_reward
            performance.append(episode_performance)
            
            # Log training progress
            if (episode + 1) % 5 == 0 or episode == 0 or episode == episodes - 1:
                print(f"Episode {episode + 1}/{episodes} - " + 
                      f"Reward: {episode_reward:.2f}, " +
                      f"Avg Position Error: {episode_performance.get('avg_pos_error', 0):.2f} km, " + 
                      f"Avg Intensity Error: {episode_performance.get('avg_intensity_error', 0):.2f} mph")
            
            # Update ensemble members periodically
            if (episode + 1) % 10 == 0:
                self._update_ensemble()
            
            # Add a small delay to prevent blocking
            await asyncio.sleep(0.01)
        
        # Store performance history
        self.training_performance = performance
        
        return performance
    
    def _update_ensemble(self):
        """Update ensemble members based on main model."""
        for member in self.ensemble_members:
            # Create a variation of the main model
            variation_factor = 0.2
            
            # Copy main model parameters with some random variation
            with torch.no_grad():
                for target_param, source_param in zip(member["q_network"].parameters(), 
                                                     self.q_network.parameters()):
                    # Add random variation to main model weights
                    noise = torch.randn_like(source_param) * variation_factor
                    target_param.copy_(source_param + noise)
            
            # Update target network
            member["target_network"].load_state_dict(member["q_network"].state_dict())
    
    def create_variation(self, variation_factor=0.3) -> 'HurricanePredictionAgent':
        """Create a slightly different version of this agent for ensemble forecasting."""
        varied_agent = HurricanePredictionAgent({
            **self.options,
            "ensemble_size": max(1, self.options["ensemble_size"] - 1)  # Smaller ensemble
        })
        
        # Copy the Q-network with variations
        with torch.no_grad():
            for target_param, source_param in zip(varied_agent.q_network.parameters(), 
                                                 self.q_network.parameters()):
                # Add random variation to weights
                noise = torch.randn_like(source_param) * variation_factor
                target_param.copy_(source_param + noise)
        
        # Update target network
        varied_agent.target_network.load_state_dict(varied_agent.q_network.state_dict())
        
        return varied_agent
    
    def get_forecast_statistics(self, predictions: List[Dict]) -> Dict:
        """Get forecast statistics including uncertainty."""
        if not predictions:
            return None
        
        # Calculate confidence intervals for each forecast time
        statistics = {}
        
        # Group predictions by forecast day
        forecast_days = list(set(p.get("day", 0) for p in predictions))
        forecast_days.sort()
        
        for day in forecast_days:
            day_predictions = [p for p in predictions if p.get("day") == day]
            
            if not day_predictions:
                continue
            
            # Calculate position statistics
            lats = [p.get("position", {}).get("lat", 0) for p in day_predictions]
            lons = [p.get("position", {}).get("lon", 0) for p in day_predictions]
            avg_lat = sum(lats) / len(lats)
            avg_lon = sum(lons) / len(lons)
            lat_std_dev = self._calculate_std_dev(lats, avg_lat)
            lon_std_dev = self._calculate_std_dev(lons, avg_lon)
            
            # Calculate intensity statistics
            wind_speeds = [p.get("wind_speed", 0) for p in day_predictions]
            avg_wind_speed = sum(wind_speeds) / len(wind_speeds)
            wind_std_dev = self._calculate_std_dev(wind_speeds, avg_wind_speed)
            
            # Calculate category ranges
            categories = []
            for p in day_predictions:
                category = p.get("category", "")
                if category == "TS" or category == "TD":
                    category_num = 0
                else:
                    try:
                        category_num = int(category)
                    except (ValueError, TypeError):
                        category_num = 0
                categories.append(category_num)
            
            min_category = min(categories)
            max_category = max(categories)
            
            # Calculate confidence based on spread
            # Lower spread = higher confidence
            position_spread = math.sqrt(lat_std_dev * lat_std_dev + lon_std_dev * lon_std_dev)
            intensity_spread = wind_std_dev
            
            # Normalize spreads (lower is better)
            normalized_position_spread = min(1, position_spread / 2)  # 2 degrees is max for normalization
            normalized_intensity_spread = min(1, intensity_spread / 40)  # 40 mph is max for normalization
            
            # Calculate confidence (100% - spread)
            position_confidence = 100 * (1 - normalized_position_spread)
            intensity_confidence = 100 * (1 - normalized_intensity_spread)
            
            # Average confidence
            overall_confidence = (position_confidence + intensity_confidence) / 2
            
            statistics[day] = {
                "position": {
                    "mean": {"lat": avg_lat, "lon": avg_lon},
                    "std_dev": {"lat": lat_std_dev, "lon": lon_std_dev},
                    "confidence": position_confidence
                },
                "intensity": {
                    "mean": avg_wind_speed,
                    "std_dev": wind_std_dev,
                    "range": [avg_wind_speed - 2*wind_std_dev, avg_wind_speed + 2*wind_std_dev],
                    "confidence": intensity_confidence
                },
                "category": {
                    "range": [min_category, max_category],
                    "mode": self._get_mode(categories)
                },
                "confidence": overall_confidence
            }
        
        return statistics
    
    def _get_mode(self, values: List) -> Any:
        """Get the most frequent value in an array."""
        if not values:
            return None
            
        counts = {}
        max_count = 0
        mode = None
        
        for value in values:
            counts[value] = counts.get(value, 0) + 1
            if counts[value] > max_count:
                max_count = counts[value]
                mode = value
        
        return mode