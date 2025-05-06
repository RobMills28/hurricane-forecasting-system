"""
Trajectory Agent for Hurricane Prediction

This agent specializes in predicting hurricane movement paths.
"""

import torch
import math
import numpy as np
import random
from typing import Dict, List

from .base_agent import BaseAgent

class TrajectoryAgent(BaseAgent):
    """Agent specialized for hurricane trajectory prediction."""
    
    def __init__(self, options: Dict = None):
        """Initialize the trajectory agent with specialized options."""
        # Default options with trajectory-specific adjustments
        trajectory_options = {
            "state_dim": 12,
            "action_dim": 9,  # 3x3 grid for lat/lon movement only
            "learning_rate": 0.0008  # Slightly lower for more stable learning
        }
        
        # Apply any provided options
        if options:
            trajectory_options.update(options)
        
        # Initialize base agent
        super().__init__(trajectory_options)
        
        # Position importance factor (emphasize position accuracy)
        self.position_weight = 0.8  # Higher weighting for position in rewards
        
        # Movement patterns for different basins
        self.movement_patterns = {
            "NA": {  # North Atlantic
                "early_stage": {"lat": 0.2, "lon": -0.6},  # Westward movement in early stage
                "mid_stage": {"lat": 0.4, "lon": -0.3},    # Start to curve northward
                "late_stage": {"lat": 0.6, "lon": 0.4}     # Recurve northeast
            },
            "WP": {  # Western Pacific
                "early_stage": {"lat": 0.1, "lon": -0.5},
                "mid_stage": {"lat": 0.3, "lon": -0.2},
                "late_stage": {"lat": 0.5, "lon": 0.3}
            },
            # Add other basins as needed
            "DEFAULT": {
                "early_stage": {"lat": 0.15, "lon": -0.5},
                "mid_stage": {"lat": 0.3, "lon": -0.3},
                "late_stage": {"lat": 0.5, "lon": 0.3}
            }
        }
    
    def predict(self, state: Dict, history: List[Dict], training: bool = False) -> Dict:
        """Make a trajectory prediction for the hurricane."""
        # Increment step counter
        self.steps += 1
        
        # Get basin-specific model if available
        basin = state.get("basin", "DEFAULT")
        model = self.basin_models.get(basin) if self.options["use_basin_models"] else None
        
        # If model is None, fall back to default model
        if model is None:
            model = self.basin_models.get("DEFAULT")
        
        # Decay epsilon for exploration
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
        
        # Convert action index to prediction (focusing only on position)
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
        """Convert discrete action index to trajectory prediction."""
        # Extract current position
        position = state.get("position", {})
        current_lat = position.get("lat", 0)
        current_lon = position.get("lon", 0)
        
        # For trajectory agent, we only care about position changes
        # We represent actions as a 3x3 grid of possible lat/lon changes
        
        # Calculate lat/lon changes based on action
        lat_action = action_idx // 3  # 0, 1, 2 for south, no change, north
        lon_action = action_idx % 3   # 0, 1, 2 for west, no change, east
        
        # Convert to changes (trajectory agent focuses only on position)
        lat_changes = [-0.5, 0.0, 0.5]  # South, no change, north
        lon_changes = [-0.5, 0.0, 0.5]  # West, no change, east
        
        lat_change = lat_changes[lat_action]
        lon_change = lon_changes[lon_action]
        
        # Get basin patterns
        basin = state.get("basin", "DEFAULT")
        patterns = self.basin_patterns.get(basin, self.basin_patterns["DEFAULT"])

        # Modify changes based on basin patterns
        if current_lat < patterns["recurve_latitude"]:
            # Early track behavior (before recurvature)
            lat_change = lat_change * 0.6 + patterns["early_lat_change"] * 0.4
            lon_change = lon_change * 0.6 + patterns["early_lon_change"] * 0.4
        else:
            # Apply recurvature effect
            recurve_effect = (current_lat - patterns["recurve_latitude"]) * patterns["recurve_strength"] / 10.0
            lat_change = lat_change + recurve_effect * 0.5
            lon_change = lon_change + recurve_effect * 0.7  # Stronger eastward component after recurvature

        # Apply changes
        new_lat = current_lat + lat_change
        new_lon = current_lon + lon_change
        
        # Ensure latitude is within bounds
        new_lat = max(-90, min(90, new_lat))
        
        # Ensure longitude is within bounds
        new_lon = ((new_lon + 180) % 360) - 180
        
        # Return position prediction only - that's this agent's focus
        return {
            "position": {"lat": new_lat, "lon": new_lon}
        }
    
    def _prediction_to_action(self, prediction):
        """Convert continuous prediction back to discrete action index."""
        # Extract predicted position change
        position = prediction.get("position", {})
        lat_change = position.get("lat", 0) - position.get("lat_prev", 0)
        lon_change = position.get("lon", 0) - position.get("lon_prev", 0)
        
        # Discretize lat change
        if lat_change <= -0.25:
            lat_action = 0  # South
        elif lat_change <= 0.25:
            lat_action = 1  # No change
        else:
            lat_action = 2  # North
        
        # Discretize lon change
        if lon_change <= -0.25:
            lon_action = 0  # West
        elif lon_change <= 0.25:
            lon_action = 1  # No change
        else:
            lon_action = 2  # East
        
        # Calculate action index
        return lat_action * 3 + lon_action
    
    def combine_ensemble_predictions(self, predictions: List[Dict], state: Dict) -> Dict:
        """Combine predictions from ensemble members, focusing on position."""
        # Extract main model prediction (last element) and ensemble predictions
        main_prediction = predictions[-1] if predictions else {}
        ensemble_predictions = predictions[:-1] if len(predictions) > 1 else []
        
        if not ensemble_predictions:
            return main_prediction
        
        # Extract position values
        positions_lat = [pred.get("position", {}).get("lat", 0) for pred in ensemble_predictions]
        positions_lon = [pred.get("position", {}).get("lon", 0) for pred in ensemble_predictions]
        
        # Get main model values
        main_lat = main_prediction.get("position", {}).get("lat", 0)
        main_lon = main_prediction.get("position", {}).get("lon", 0)
        
        # Add main model to ensemble values with higher weight
        main_weight = 2.0  # Main model has 2x the weight of ensemble members
        
        # Add weighted main model values
        weighted_positions_lat = positions_lat + [main_lat] * int(main_weight)
        weighted_positions_lon = positions_lon + [main_lon] * int(main_weight)
        
        # Remove outliers (values beyond 2 standard deviations)
        filtered_lat = self._filter_outliers(weighted_positions_lat)
        filtered_lon = self._filter_outliers(weighted_positions_lon)
        
        # Calculate means with outliers removed
        avg_lat = sum(filtered_lat) / len(filtered_lat) if filtered_lat else main_lat
        avg_lon = sum(filtered_lon) / len(filtered_lon) if filtered_lon else main_lon
        
        # Calculate standard deviations for uncertainty estimation
        lat_std_dev = self._calculate_std_dev(filtered_lat, avg_lat)
        lon_std_dev = self._calculate_std_dev(filtered_lon, avg_lon)
        
        return {
            "position": {"lat": avg_lat, "lon": avg_lon},
            "uncertainty": {
                "position": {"lat": lat_std_dev, "lon": lon_std_dev}
            }
        }