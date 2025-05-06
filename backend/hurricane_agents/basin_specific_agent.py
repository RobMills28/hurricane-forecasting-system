"""
Basin-Specific Agent for Hurricane Prediction

This agent specializes in a specific ocean basin's hurricane patterns.
"""

import torch
import math
import numpy as np
import random
from typing import Dict, List

from .base_agent import BaseAgent

class BasinSpecificAgent(BaseAgent):
    """Agent specialized for a specific ocean basin."""
    
    def __init__(self, basin: str, options: Dict = None):
        """
        Initialize the basin-specific agent.
        
        Args:
            basin: Ocean basin code (NA, WP, etc.)
            options: Configuration options
        """
        # Default options with basin-specific adjustments
        basin_options = {
            "state_dim": 12,
            "action_dim": 15,  # Full action space
            "learning_rate": 0.001
        }
        
        # Apply any provided options
        if options:
            basin_options.update(options)
        
        # Initialize base agent
        super().__init__(basin_options)
        
        # Store target basin
        self.target_basin = basin
        
        # Set basin-specific parameters
        if basin == "NA":  # North Atlantic
            self.basin_factors = {
                "recurvature_importance": 0.7,  # Higher importance for recurvature
                "sst_influence": 0.8,           # Higher SST influence
                "early_intensification": 0.6    # Medium early intensification
            }
        elif basin == "WP":  # Western Pacific
            self.basin_factors = {
                "recurvature_importance": 0.5,  # Medium importance for recurvature
                "sst_influence": 0.7,           # Medium-high SST influence
                "early_intensification": 0.8    # Higher early intensification
            }
        elif basin == "EP":  # Eastern Pacific
            self.basin_factors = {
                "recurvature_importance": 0.3,  # Lower importance for recurvature
                "sst_influence": 0.8,           # Higher SST influence
                "early_intensification": 0.7    # Medium-high early intensification
            }
        else:  # Default basin
            self.basin_factors = {
                "recurvature_importance": 0.5,  # Medium importance for recurvature
                "sst_influence": 0.6,           # Medium SST influence
                "early_intensification": 0.6    # Medium early intensification
            }
    
    def predict(self, state: Dict, history: List[Dict], training: bool = False) -> Dict:
        """
        Make a prediction for hurricane in the specific basin.
        If the hurricane is not in this agent's basin, returns a default prediction.
        """
        # Check if the hurricane is in this agent's basin
        basin = state.get("basin", "DEFAULT")
        if basin != self.target_basin and training is False:
            # This is not this agent's specialty, return a minimal prediction
            # Ensemble will ignore or minimize this agent's contribution
            return {
                "position": state.get("position", {}),
                "wind_speed": state.get("wind_speed", 0),
                "pressure": state.get("pressure", 1010),
                "confidence": 0.1  # Very low confidence for out-of-basin prediction
            }
        
        # Increment step counter
        self.steps += 1
        
        # For basin-specific agent, only use the model for this basin
        model = self.basin_models.get(self.target_basin)
        
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
                q_values = model["q_network"](state_tensor)
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
            combined = self.combine_ensemble_predictions(ensemble_predictions, state)
            
            # For basin-specific agents, add confidence based on basin match
            combined["confidence"] = 1.0 if basin == self.target_basin else 0.1
            
            return combined
        
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
        """Convert discrete action index to hurricane prediction with basin-specific factors."""
        # Extract current position and intensity
        position = state.get("position", {})
        current_lat = position.get("lat", 0)
        current_lon = position.get("lon", 0)
        current_wind = state.get("wind_speed", 0)
        current_pressure = state.get("pressure", 1010)
        
        # Decode action - similar to original agent but with basin-specific adjustments
        # We have 15 actions: 5 latitude x 3 longitude changes
        lat_action = action_idx // 3  # 0-4
        lon_action = (action_idx % 3)  # 0-2
        
        # Convert actions to changes
        lat_changes = [-1.0, -0.5, 0.0, 0.5, 1.0]
        lon_changes = [-0.5, 0.0, 0.5]
        
        lat_change = lat_changes[lat_action]
        lon_change = lon_changes[lon_action]
        
        # Get basin patterns with added basin-specific factors
        patterns = self.basin_patterns.get(self.target_basin, self.basin_patterns["DEFAULT"])

        # Modify changes based on basin patterns, but enhanced with basin-specific factors
        recurvature_factor = self.basin_factors["recurvature_importance"]
        
        if current_lat < patterns["recurve_latitude"]:
            # Early track behavior (before recurvature)
            lat_change = lat_change * 0.7 + patterns["early_lat_change"] * 0.3
            lon_change = lon_change * 0.7 + patterns["early_lon_change"] * 0.3
        else:
            # Apply recurvature effect with basin-specific factor
            recurve_effect = (current_lat - patterns["recurve_latitude"]) * patterns["recurve_strength"] / 10.0
            lat_change = lat_change + recurve_effect * 0.4 * recurvature_factor
            lon_change = lon_change + recurve_effect * 0.6 * recurvature_factor
        
        # Apply changes
        new_lat = current_lat + lat_change
        new_lon = current_lon + lon_change
        
        # Ensure latitude is within bounds
        new_lat = max(-90, min(90, new_lat))
        
        # Ensure longitude is within bounds
        new_lon = ((new_lon + 180) % 360) - 180
        
        # Calculate intensity change based on basin-specific factors
        sea_surface_temp = state.get("sea_surface_temp", {}).get("value", 28)
        sst_effect = self.calculate_sst_effect(sea_surface_temp) * self.basin_factors["sst_influence"]
        
        # Base intensity change
        intensity_change = 0
        
        # Apply SST effect with basin-specific factor
        intensity_change += sst_effect * 10  # Scale SST effect to reasonable intensity change
        
        # Apply early intensification factor
        if current_wind < 100:  # Only apply to developing storms
            intensity_change += self.basin_factors["early_intensification"] * 5
        
        # Calculate new intensity
        new_wind = max(0, current_wind + intensity_change)
        
        # Calculate pressure based on wind speed
        new_pressure = 1010 - (new_wind / 1.15)**2 / 100
        
        # Constrain pressure to realistic values
        new_pressure = max(880, min(1020, new_pressure))
        
        return {
            "position": {"lat": new_lat, "lon": new_lon},
            "wind_speed": new_wind,
            "pressure": new_pressure
        }
    
    def _prediction_to_action(self, prediction):
        """Convert continuous prediction back to discrete action index."""
        # Extract prediction values
        position = prediction.get("position", {})
        lat_change = position.get("lat", 0) - position.get("lat_prev", 0)
        lon_change = position.get("lon", 0) - position.get("lon_prev", 0)
        
        # Discretize lat change
        if lat_change <= -0.75:
            lat_action = 0  # Large negative
        elif lat_change <= -0.25:
            lat_action = 1  # Small negative
        elif lat_change <= 0.25:
            lat_action = 2  # Neutral
        elif lat_change <= 0.75:
            lat_action = 3  # Small positive
        else:
            lat_action = 4  # Large positive
        
        # Discretize lon change
        if lon_change <= -0.25:
            lon_action = 0  # Negative
        elif lon_change <= 0.25:
            lon_action = 1  # Neutral
        else:
            lon_action = 2  # Positive
        
        # Calculate action index
        return lat_action * 3 + lon_action
    
    def combine_ensemble_predictions(self, predictions: List[Dict], state: Dict) -> Dict:
        """Combine predictions from ensemble members, incorporating basin-specific knowledge."""
        # Extract main model prediction (last element) and ensemble predictions
        main_prediction = predictions[-1] if predictions else {}
        ensemble_predictions = predictions[:-1] if len(predictions) > 1 else []
        
        if not ensemble_predictions:
            return main_prediction
        
        # Extract values
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