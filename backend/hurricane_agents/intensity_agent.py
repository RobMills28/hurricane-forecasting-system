"""
Intensity Agent for Hurricane Prediction

This agent specializes in predicting hurricane intensity (wind speed and pressure).
"""

import torch
import math
import numpy as np
import random
from typing import Dict, List

from .base_agent import BaseAgent

class IntensityAgent(BaseAgent):
    """Agent specialized for hurricane intensity prediction."""
    
    def __init__(self, options: Dict = None):
        """Initialize the intensity agent with specialized options."""
        # Default options with intensity-specific adjustments
        intensity_options = {
            "state_dim": 12,
            "action_dim": 6,  # Simplified action space focused on intensity
            "learning_rate": 0.001
        }
        
        # Apply any provided options
        if options:
            intensity_options.update(options)
        
        # Initialize base agent
        super().__init__(intensity_options)
        
        # Intensity importance factor
        self.intensity_weight = 0.9  # Higher weighting for intensity in rewards
        
        # Enhanced environmental relationships
        self.enhanced_sst_effects = True  # Use enhanced SST modeling
        self.rapid_intensification_threshold = 30  # mph/24hr
    
    def predict(self, state: Dict, history: List[Dict], training: bool = False) -> Dict:
        """Make an intensity prediction for the hurricane."""
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
        
        # Preprocess state with intensity focus
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
        
        # Convert action index to prediction (focusing only on intensity)
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
        """Convert discrete action index to intensity prediction."""
        # Extract current intensity values
        current_wind = state.get("wind_speed", 0)
        current_pressure = state.get("pressure", 1010)
        
        # For intensity agent, we care about wind speed and pressure changes
        # We represent actions as intensity change options
        
        # Calculate intensity changes based on action
        # 0: major decrease, 1: slight decrease, 2: no change, 
        # 3: slight increase, 4: moderate increase, 5: major increase
        intensity_changes = [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
        intensity_change = intensity_changes[action_idx]
        
        # Get basin patterns for intensity factors
        basin = state.get("basin", "DEFAULT")
        patterns = self.basin_patterns.get(basin, self.basin_patterns["DEFAULT"])

        # Modify intensity change based on basin patterns and environmental factors
        intensity_change = intensity_change * (1 + patterns["intensity_change_rate"])
        
        # Apply environmental factors more strongly for the intensity agent
        position = state.get("position", {})
        lat = position.get("lat", 0)
        
        # Apply sea surface temperature effect with higher importance
        sea_surface_temp = state.get("sea_surface_temp", {})
        if sea_surface_temp and "value" in sea_surface_temp:
            # SST effect on intensity with higher weight (1.5x)
            sst_effect = self.calculate_sst_effect(sea_surface_temp["value"]) * 1.5
            intensity_change = intensity_change * (1 + sst_effect)
        
        # Apply latitude effect with focus on intensity
        latitude_effect = self.calculate_latitude_effect(lat, basin)
        if abs(lat) > 25:
            # Weakening at higher latitudes affects intensity more
            intensity_change = intensity_change * (1 - (latitude_effect * 0.15))
        
        # Calculate new intensity values
        new_wind = max(0, current_wind + intensity_change)
        
        # Calculate pressure using enhanced wind-pressure relationship
        # P = 1010 - (wind_speed/1.15)^2/100
        new_pressure = 1010 - (new_wind / 1.15)**2 / 100
        
        # Constrain pressure to realistic values
        new_pressure = max(880, min(1020, new_pressure))
        
        # Return intensity prediction only - that's this agent's focus
        return {
            "wind_speed": new_wind,
            "pressure": new_pressure
        }
    
    def _prediction_to_action(self, prediction):
        """Convert continuous prediction back to discrete action index."""
        # Extract predicted intensity change
        intensity_change = prediction.get("wind_speed", 0) - prediction.get("wind_speed_prev", 0)
        
        # Discretize intensity change into action
        if intensity_change <= -15:
            return 0  # Major decrease
        elif intensity_change <= -5:
            return 1  # Slight decrease
        elif intensity_change <= 5:
            return 2  # No change
        elif intensity_change <= 15:
            return 3  # Slight increase
        elif intensity_change <= 25:
            return 4  # Moderate increase
        else:
            return 5  # Major increase
    
    def combine_ensemble_predictions(self, predictions: List[Dict], state: Dict) -> Dict:
        """Combine predictions from ensemble members, focusing on intensity."""
        # Extract main model prediction (last element) and ensemble predictions
        main_prediction = predictions[-1] if predictions else {}
        ensemble_predictions = predictions[:-1] if len(predictions) > 1 else []
        
        if not ensemble_predictions:
            return main_prediction
        
        # Extract intensity values
        wind_speeds = [pred.get("wind_speed", 0) for pred in ensemble_predictions]
        pressures = [pred.get("pressure", 0) for pred in ensemble_predictions]
        
        # Get main model values
        main_wind = main_prediction.get("wind_speed", 0)
        main_pressure = main_prediction.get("pressure", 0)
        
        # Add main model to ensemble values with higher weight
        main_weight = 2.0  # Main model has 2x the weight of ensemble members
        
        # Add weighted main model values
        weighted_wind_speeds = wind_speeds + [main_wind] * int(main_weight)
        weighted_pressures = pressures + [main_pressure] * int(main_weight)
        
        # Remove outliers (values beyond 2 standard deviations)
        filtered_wind = self._filter_outliers(weighted_wind_speeds)
        filtered_pressure = self._filter_outliers(weighted_pressures)
        
        # Calculate means with outliers removed
        avg_wind = sum(filtered_wind) / len(filtered_wind) if filtered_wind else main_wind
        avg_pressure = sum(filtered_pressure) / len(filtered_pressure) if filtered_pressure else main_pressure
        
        # Calculate standard deviations for uncertainty estimation
        wind_std_dev = self._calculate_std_dev(filtered_wind, avg_wind)
        pressure_std_dev = self._calculate_std_dev(filtered_pressure, avg_pressure)
        
        return {
            "wind_speed": avg_wind,
            "pressure": avg_pressure,
            "uncertainty": {
                "wind_speed": wind_std_dev,
                "pressure": pressure_std_dev
            }
        }