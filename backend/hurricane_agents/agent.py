"""
Hurricane Prediction Agent

This module provides agent implementations for hurricane trajectory and intensity prediction
using ensemble methods and basin-specific models.
"""

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Any, Union

class HurricanePredictionAgent:
    """
    Advanced Hurricane Prediction Agent using ensemble methods
    and basin-specific models.
    
    This is the Python equivalent of the JavaScript HurricanePredictionAgent class.
    """
    
    def __init__(self, options: Dict = None):
        """
        Initialize the hurricane prediction agent.
        
        Args:
            options: Configuration options for the agent
        """
        # Configuration options
        self.options = {
            "use_dynamic_weights": True,    # Use dynamic weights based on storm phase
            "use_basin_models": True,       # Use basin-specific models
            "ensemble_size": 5,            # Number of sub-models in ensemble
        }
        
        if options:
            self.options.update(options)
        
        # Initialize weights for trajectory prediction
        self.weights = {
            "position": {
                "lat": [0.1] * 5,  # Weights for previous 5 positions
                "lon": [0.1] * 5
            },
            "intensity": [0.1] * 5,
            "pressure": [0.1] * 5
        }
        
        # Basin-specific models
        self.basin_models = {
            "NA": self.create_initial_weights().copy(),  # North Atlantic
            "EP": self.create_initial_weights().copy(),  # Eastern Pacific
            "WP": self.create_initial_weights().copy(),  # Western Pacific
            "NI": self.create_initial_weights().copy(),  # North Indian
            "SI": self.create_initial_weights().copy(),  # South Indian
            "SP": self.create_initial_weights().copy(),  # South Pacific
            "DEFAULT": self.create_initial_weights().copy()  # Default model
        }
        
        # Phase-specific weight adjustments
        self.phase_adjustments = {
            "early": {
                "position": {"factor": 1.2},     # Early phase: directional persistence is stronger
                "intensity": {"factor": 1.5}     # Early phase: rapid intensification possible
            },
            "peak": {
                "position": {"factor": 1.0},     # Peak phase: more balanced
                "intensity": {"factor": 0.8}     # Peak phase: intensity changes less dramatically
            },
            "late": {
                "position": {"factor": 0.7},     # Late phase: track becomes more variable
                "intensity": {"factor": 1.3}     # Late phase: rapid weakening possible
            }
        }
        
        # Environmental factor weights
        self.environmental_factors = {
            "sea_surface_temp": 0.3,              # Influence of SST on intensification
            "latitudinal_effect": 0.2,           # Higher latitudes encourage weakening/recurvature
            "seasonal_effect": 0.1               # Seasonal patterns
        }
        
        self.learning_rate = 0.01
        self.ensemble_members = self.initialize_ensemble()
        
        # Performance tracking
        self.training_performance = []
    
    def create_initial_weights(self) -> Dict:
        """
        Create initial weights object with slight variations for ensembles.
        
        Returns:
            Dictionary of initial weights
        """
        return {
            "position": {
                "lat": [0.1 + (random.random() * 0.05 - 0.025) for _ in range(5)],
                "lon": [0.1 + (random.random() * 0.05 - 0.025) for _ in range(5)]
            },
            "intensity": [0.1 + (random.random() * 0.05 - 0.025) for _ in range(5)],
            "pressure": [0.1 + (random.random() * 0.05 - 0.025) for _ in range(5)]
        }
    
    def initialize_ensemble(self) -> List[Dict]:
        """
        Initialize ensemble models.
        Creates slightly different variations of the model for ensemble forecasting.
        
        Returns:
            List of ensemble member configurations
        """
        ensemble = []
        
        for i in range(self.options["ensemble_size"]):
            # Create variation of weights
            variation_factor = 0.1 + (random.random() * 0.1)
            weights = {}
            
            # Deep copy with variations
            # Position weights
            weights["position"] = {
                "lat": [w * (1 + (random.random() * variation_factor * 2 - variation_factor)) 
                        for w in self.weights["position"]["lat"]],
                "lon": [w * (1 + (random.random() * variation_factor * 2 - variation_factor)) 
                        for w in self.weights["position"]["lon"]]
            }
            
            # Intensity weights
            weights["intensity"] = [w * (1 + (random.random() * variation_factor * 2 - variation_factor)) 
                                  for w in self.weights["intensity"]]
            
            # Pressure weights
            weights["pressure"] = [w * (1 + (random.random() * variation_factor * 2 - variation_factor)) 
                                 for w in self.weights["pressure"]]
            
            ensemble.append({
                "weights": weights,
                "environmental_factors": self.environmental_factors.copy(),
                "learning_rate": self.learning_rate * (1 + (random.random() * 0.2 - 0.1))
            })
        
        return ensemble
    
    def predict(self, state: Dict, history: List[Dict]) -> Dict:
        """
        Make a prediction based on current state and history.
        
        Args:
            state: Current hurricane state
            history: List of historical states
                
        Returns:
            Prediction dictionary with position, wind speed, and pressure
        """
        # Get basin-specific model if available
        basin = state.get("basin", "DEFAULT")
        weights = self.basin_models.get(basin) if self.options["use_basin_models"] else self.weights
        
        # If weights is None, fall back to default weights
        if weights is None:
            weights = self.basin_models.get("DEFAULT")
            if weights is None:  # If DEFAULT is also missing, use self.weights
                weights = self.weights
        
        # Make individual ensemble predictions
        ensemble_predictions = [
            self.make_single_prediction(
                state, 
                history, 
                member["weights"], 
                member["environmental_factors"]
            )
            for member in self.ensemble_members
        ]
        
        # Add main model prediction
        ensemble_predictions.append(
            self.make_single_prediction(state, history, weights, self.environmental_factors)
        )
        
        # Combine ensemble predictions
        return self.combine_ensemble_predictions(ensemble_predictions, state)
    
    def make_single_prediction(self, state: Dict, history: List[Dict], 
                              weights: Dict, environmental_factors: Dict) -> Dict:
        """
        Make a single model prediction.
        
        Args:
            state: Current hurricane state
            history: List of historical states
            weights: Weights for this prediction
            environmental_factors: Environmental factor weights
            
        Returns:
            Single prediction dictionary
        """
        # Get storm phase based on history length
        storm_phase = self.determine_storm_phase(history)
        
        # Extract the last positions (or fewer if not available)
        positions = [h.get("state", {}).get("position", {}) for h in history[-5:]]
        positions.reverse()  # Oldest first
        
        # Add current position
        current_position = state.get("position", {})
        positions.append(current_position)
        
        # Extract the last wind speeds
        wind_speeds = [h.get("state", {}).get("wind_speed", 0) for h in history[-5:]]
        wind_speeds.reverse()  # Oldest first
        
        # Add current wind speed
        current_wind = state.get("wind_speed", 0)
        wind_speeds.append(current_wind)
        
        # Extract the last pressures
        pressures = [h.get("state", {}).get("pressure", 1000) for h in history[-5:]]
        pressures.reverse()  # Oldest first
        
        # Add current pressure
        current_pressure = state.get("pressure", 1000)
        pressures.append(current_pressure)
        
        # Adjust weights based on storm phase if enabled
        adjusted_weights = weights.copy()
        if self.options["use_dynamic_weights"] and storm_phase:
            adjusted_weights = self.adjust_weights_for_phase(weights, storm_phase)
        
        # Predict next position using weighted average of position changes
        lat_predict = current_position.get("lat", 0)
        lon_predict = current_position.get("lon", 0)
        
        # Calculate position changes and apply weights
        for i in range(1, len(positions)):
            idx = i - 1
            if idx < len(adjusted_weights["position"]["lat"]):
                pos_i = positions[i]
                pos_i_prev = positions[i-1]
                
                # Safely extract lat/lon values
                lat_i = pos_i.get("lat", 0)
                lat_i_prev = pos_i_prev.get("lat", 0)
                lon_i = pos_i.get("lon", 0)
                lon_i_prev = pos_i_prev.get("lon", 0)
                
                lat_predict += (lat_i - lat_i_prev) * adjusted_weights["position"]["lat"][idx]
                lon_predict += (lon_i - lon_i_prev) * adjusted_weights["position"]["lon"][idx]
        
        # Predict intensity (wind speed)
        wind_predict = current_wind
        for i in range(1, len(wind_speeds)):
            idx = i - 1
            if idx < len(adjusted_weights["intensity"]):
                wind_predict += (wind_speeds[i] - wind_speeds[i-1]) * adjusted_weights["intensity"][idx]
        
        # Predict pressure
        pressure_predict = current_pressure
        for i in range(1, len(pressures)):
            idx = i - 1
            if idx < len(adjusted_weights["pressure"]):
                pressure_predict += (pressures[i] - pressures[i-1]) * adjusted_weights["pressure"][idx]
        
        # Apply environmental factors
        # Sea surface temperature effect
        sea_surface_temp = state.get("sea_surface_temp", {})
        if sea_surface_temp and "value" in sea_surface_temp:
            # SST effect on intensity
            sst_effect = self.calculate_sst_effect(sea_surface_temp["value"])
            wind_predict *= (1 + (sst_effect * environmental_factors["sea_surface_temp"]))
            
            # Higher SST typically means lower pressure
            pressure_predict *= (1 - (sst_effect * environmental_factors["sea_surface_temp"] * 0.05))
        
        # Latitude effect (higher latitudes typically mean weakening and recurvature)
        latitude = current_position.get("lat", 0)
        latitude_effect = self.calculate_latitude_effect(latitude, state.get("basin"))
        
        # Apply latitude effect to track and intensity
        if abs(latitude) > 25:
            # More poleward movement at higher latitudes
            poleward_direction = 1 if latitude > 0 else -1
            lat_predict += latitude_effect * poleward_direction * environmental_factors["latitudinal_effect"]
            
            # Generally eastward movement at higher latitudes
            lon_predict += latitude_effect * 0.2 * environmental_factors["latitudinal_effect"]
            
            # Weakening at higher latitudes
            wind_predict *= (1 - (latitude_effect * environmental_factors["latitudinal_effect"] * 0.1))
            pressure_predict *= (1 + (latitude_effect * environmental_factors["latitudinal_effect"] * 0.02))
        
        # Limit pressure to realistic values
        pressure_predict = max(880, min(1020, pressure_predict))
        
        # Ensure wind speed and pressure are physically consistent
        # Lower pressure should correlate with higher wind speed
        pressure_wind_correlation = self.correlate_wind_and_pressure(wind_predict, pressure_predict)
        wind_predict = pressure_wind_correlation["wind_speed"]
        pressure_predict = pressure_wind_correlation["pressure"]
        
        return {
            "position": {"lat": lat_predict, "lon": lon_predict},
            "wind_speed": wind_predict,
            "pressure": pressure_predict
        }
    
    def determine_storm_phase(self, history: List[Dict]) -> str:
        """
        Determine the current phase of the storm based on history.
        
        Args:
            history: List of historical states
            
        Returns:
            Storm phase ("early", "peak", or "late")
        """
        history_length = len(history)
        if history_length < 4:
            return "early"
        if history_length > 12:
            return "late"
        return "peak"
    
    def adjust_weights_for_phase(self, weights: Dict, phase: str) -> Dict:
        """
        Adjust weights based on storm phase.
        
        Args:
            weights: Original weights
            phase: Storm phase ("early", "peak", or "late")
            
        Returns:
            Adjusted weights
        """
        phase_adjustment = self.phase_adjustments.get(phase, self.phase_adjustments["peak"])
        adjusted_weights = weights.copy()
        
        # Adjust position weights
        if "position" in phase_adjustment and "factor" in phase_adjustment["position"]:
            factor = phase_adjustment["position"]["factor"]
            adjusted_weights["position"] = {
                "lat": [w * factor for w in weights["position"]["lat"]],
                "lon": [w * factor for w in weights["position"]["lon"]]
            }
        
        # Adjust intensity weights
        if "intensity" in phase_adjustment and "factor" in phase_adjustment["intensity"]:
            factor = phase_adjustment["intensity"]["factor"]
            adjusted_weights["intensity"] = [w * factor for w in weights["intensity"]]
        
        return adjusted_weights
    
    def calculate_sst_effect(self, sst: float) -> float:
        """
        Calculate the effect of sea surface temperature on hurricane intensification.
        
        Args:
            sst: Sea surface temperature in degrees Celsius
            
        Returns:
            Effect multiplier
        """
        # Hurricanes typically intensify over waters warmer than 26°C
        # and intensify rapidly over waters warmer than 28°C
        if sst < 26:
            return -0.05  # Slight weakening effect
        if sst < 28:
            return 0.02   # Slight intensification
        if sst < 30:
            return 0.05   # Moderate intensification
        return 0.08       # Strong intensification
    
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
    
    def correlate_wind_and_pressure(self, wind_speed: float, pressure: float) -> Dict:
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
        ideal_wind = 1.15 * math.sqrt((1010 - pressure) * 100)
        
        # Average the current prediction with the "ideal" values
        return {
            "wind_speed": (wind_speed + ideal_wind) / 2,
            "pressure": (pressure + ideal_pressure) / 2
        }
    
    def combine_ensemble_predictions(self, predictions: List[Dict], state: Dict) -> Dict:
        """
        Combine predictions from ensemble members.
        
        Args:
            predictions: List of individual predictions
            state: Current state
            
        Returns:
            Combined prediction with uncertainty estimates
        """
        # Calculate weighted average of predictions
        # More weight is given to the main model
        
        # Simple arithmetic mean for position
        total_lat = sum(pred.get("position", {}).get("lat", 0) for pred in predictions)
        total_lon = sum(pred.get("position", {}).get("lon", 0) for pred in predictions)
        avg_lat = total_lat / len(predictions)
        avg_lon = total_lon / len(predictions)
        
        # Simple arithmetic mean for intensity
        total_wind = sum(pred.get("wind_speed", 0) for pred in predictions)
        total_pressure = sum(pred.get("pressure", 0) for pred in predictions)
        avg_wind = total_wind / len(predictions)
        avg_pressure = total_pressure / len(predictions)
        
        # Calculate standard deviations for uncertainty estimation
        lat_std_dev = self.calculate_std_dev(
            [pred.get("position", {}).get("lat", 0) for pred in predictions], 
            avg_lat
        )
        lon_std_dev = self.calculate_std_dev(
            [pred.get("position", {}).get("lon", 0) for pred in predictions], 
            avg_lon
        )
        wind_std_dev = self.calculate_std_dev(
            [pred.get("wind_speed", 0) for pred in predictions], 
            avg_wind
        )
        pressure_std_dev = self.calculate_std_dev(
            [pred.get("pressure", 0) for pred in predictions], 
            avg_pressure
        )
        
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
    
    def calculate_std_dev(self, values: List[float], mean: float) -> float:
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
    
    def update(self, prediction: Dict, actual: Dict, reward: float) -> None:
        """
        Update the model based on reward and actual outcomes.
        
        Args:
            prediction: The predicted values
            actual: The actual values
            reward: The reward received
        """
        if not actual:
            return
        
        # Update position weights
        pred_position = prediction.get("position", {})
        actual_position = actual.get("position", {})
        
        lat_error = actual_position.get("lat", 0) - pred_position.get("lat", 0)
        lon_error = actual_position.get("lon", 0) - pred_position.get("lon", 0)
        
        for i in range(len(self.weights["position"]["lat"])):
            self.weights["position"]["lat"][i] += self.learning_rate * lat_error * reward / 100
            self.weights["position"]["lon"][i] += self.learning_rate * lon_error * reward / 100
        
        # Update intensity weights
        intensity_error = actual.get("wind_speed", 0) - prediction.get("wind_speed", 0)
        for i in range(len(self.weights["intensity"])):
            self.weights["intensity"][i] += self.learning_rate * intensity_error * reward / 100
        
        # Update pressure weights
        pressure_error = actual.get("pressure", 0) - prediction.get("pressure", 0)
        for i in range(len(self.weights["pressure"])):
            self.weights["pressure"][i] += self.learning_rate * pressure_error * reward / 100
        
        # If using basin-specific models, update those too
        if self.options["use_basin_models"] and "basin" in actual:
            basin = actual["basin"]
            if basin in self.basin_models:
                basin_model = self.basin_models[basin]
                
                # Higher learning rate for basin-specific models
                basin_lr_multiplier = 1.5
                
                for i in range(len(basin_model["position"]["lat"])):
                    basin_model["position"]["lat"][i] += (
                        self.learning_rate * lat_error * reward / 100 * basin_lr_multiplier
                    )
                    basin_model["position"]["lon"][i] += (
                        self.learning_rate * lon_error * reward / 100 * basin_lr_multiplier
                    )
                
                for i in range(len(basin_model["intensity"])):
                    basin_model["intensity"][i] += (
                        self.learning_rate * intensity_error * reward / 100 * basin_lr_multiplier
                    )
                
                for i in range(len(basin_model["pressure"])):
                    basin_model["pressure"][i] += (
                        self.learning_rate * pressure_error * reward / 100 * basin_lr_multiplier
                    )
        
        # Update ensemble members (with smaller updates)
        for member in self.ensemble_members:
            member_weights = member["weights"]
            member_lr = member["learning_rate"]
            
            # Smaller learning rate for ensemble members
            ensemble_lr_multiplier = 0.5
            
            for i in range(len(member_weights["position"]["lat"])):
                member_weights["position"]["lat"][i] += (
                    member_lr * lat_error * reward / 100 * ensemble_lr_multiplier
                )
                member_weights["position"]["lon"][i] += (
                    member_lr * lon_error * reward / 100 * ensemble_lr_multiplier
                )
            
            for i in range(len(member_weights["intensity"])):
                member_weights["intensity"][i] += (
                    member_lr * intensity_error * reward / 100 * ensemble_lr_multiplier
                )
            
            for i in range(len(member_weights["pressure"])):
                member_weights["pressure"][i] += (
                    member_lr * pressure_error * reward / 100 * ensemble_lr_multiplier
                )
    
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
            history = []
            
            # Episode loop
            while not done:
                # Get current state
                current_state = environment.get_state()
                
                # Make prediction
                action = self.predict(current_state, history)
                
                # Take action and observe result
                result = environment.step(action)
                next_state = result.get("state", {})
                reward = result.get("reward", 0)
                done = result.get("done", False)
                
                # Store state for history
                history.append({"state": current_state, "action": action})
                
                # Update model
                actual = environment.get_actual_for_time_step(environment.time_step)
                self.update(action, actual, reward)
                
            # Evaluate performance
            episode_performance = environment.evaluate_performance()
            performance.append(episode_performance)
            
            # Log training progress
            if (episode + 1) % 5 == 0 or episode == 0 or episode == episodes - 1:
                print(f"Episode {episode + 1}/{episodes} - " + 
                      f"Avg Position Error: {episode_performance.get('avg_pos_error', 0):.2f} km, " + 
                      f"Avg Intensity Error: {episode_performance.get('avg_intensity_error', 0):.2f} mph")
        
        # Store performance history
        self.training_performance = performance
        
        return performance
    
    def create_variation(self, variation_factor=0.3) -> 'HurricanePredictionAgent':
        """
        Create a slightly different version of this agent for ensemble forecasting.
        
        Args:
            variation_factor: Factor to control the amount of variation
            
        Returns:
            New agent with varied parameters
        """
        varied_agent = HurricanePredictionAgent({
            **self.options,
            "ensemble_size": max(1, self.options["ensemble_size"] - 1)  # Smaller ensemble to save computation
        })
        
        # Vary position weights
        for i in range(len(self.weights["position"]["lat"])):
            varied_agent.weights["position"]["lat"][i] = self.weights["position"]["lat"][i] * (
                1 + (random.random() * variation_factor * 2 - variation_factor)
            )
            varied_agent.weights["position"]["lon"][i] = self.weights["position"]["lon"][i] * (
                1 + (random.random() * variation_factor * 2 - variation_factor)
            )
        
        # Vary intensity weights
        for i in range(len(self.weights["intensity"])):
            varied_agent.weights["intensity"][i] = self.weights["intensity"][i] * (
                1 + (random.random() * variation_factor * 2 - variation_factor)
            )
        
        # Vary pressure weights
        for i in range(len(self.weights["pressure"])):
            varied_agent.weights["pressure"][i] = self.weights["pressure"][i] * (
                1 + (random.random() * variation_factor * 2 - variation_factor)
            )
        
        # Vary environmental factors
        for factor in self.environmental_factors:
            varied_agent.environmental_factors[factor] = self.environmental_factors[factor] * (
                1 + (random.random() * variation_factor - variation_factor/2)
            )
        
        # Vary learning rate
        varied_agent.learning_rate = self.learning_rate * (
            1 + (random.random() * 0.4 - 0.2)
        )
        
        return varied_agent
    
    def get_forecast_statistics(self, predictions: List[Dict]) -> Dict:
        """
        Get forecast statistics including uncertainty.
        
        Args:
            predictions: List of prediction points
            
        Returns:
            Dictionary of forecast statistics
        """
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
            lat_std_dev = self.calculate_std_dev(lats, avg_lat)
            lon_std_dev = self.calculate_std_dev(lons, avg_lon)
            
            # Calculate intensity statistics
            wind_speeds = [p.get("wind_speed", 0) for p in day_predictions]
            avg_wind_speed = sum(wind_speeds) / len(wind_speeds)
            wind_std_dev = self.calculate_std_dev(wind_speeds, avg_wind_speed)
            
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
                    "mode": self.get_mode(categories)
                },
                "confidence": overall_confidence
            }
        
        return statistics
    
    def get_mode(self, values: List) -> Any:
        """
        Get the most frequent value in an array.
        
        Args:
            values: List of values
            
        Returns:
            Most frequent value
        """
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