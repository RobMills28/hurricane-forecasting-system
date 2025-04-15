"""
Hurricane Prediction Environment

This module provides an environment for hurricane trajectory and intensity prediction
using real-world data and sophisticated modeling techniques.
"""

import numpy as np
import pandas as pd
import random
import math
from typing import Dict, List, Tuple, Optional, Any, Union


class HurricaneEnvironment:
    """
    Environment for hurricane prediction that simulates hurricane development
    and provides an interface for agents to interact with.
    
    This is the Python equivalent of the JavaScript HurricaneEnvironment class.
    """
    
    def __init__(self):
        """Initialize the hurricane environment."""
        self.hurricane_data = []  # Historical hurricane data
        self.nasa_data = {}       # NASA data layers (sea surface temp, etc.)
        self.current_state = None
        self.time_step = 0        # Current time step in the simulation
        self.history = []         # Track agent predictions for evaluation
        self.basin_models = {}    # Basin-specific models for regional specialization
    
    async def initialize(self, historical_data: List[Dict], nasa_data_service: Dict) -> Dict:
        """
        Initialize the environment with data.
        
        Args:
            historical_data: List of historical hurricane data
            nasa_data_service: Service for accessing NASA data
            
        Returns:
            The initial state
        """
        self.hurricane_data = historical_data
        self.nasa_data = nasa_data_service
        
        # Organize data by basin for basin-specific training
        self.organize_data_by_basin()
        
        self.reset()
        return self.get_state()
    
    def organize_data_by_basin(self) -> None:
        """Organize hurricane data by basin for specialized training."""
        basin_data = {}
        
        for hurricane in self.hurricane_data:
            basin = hurricane.get('basin', 'UNKNOWN')
            if basin not in basin_data:
                basin_data[basin] = []
            basin_data[basin].append(hurricane)
        
        self.basin_data = basin_data
        
        # Log statistics about data distribution
        basin_stats = ", ".join([f"{basin}: {len(data)} storms" 
                               for basin, data in basin_data.items()])
        print(f"Hurricane data by basin: {basin_stats}")
    
    def reset(self) -> Dict:
        """
        Reset the environment to initial state for a new episode.
        
        Returns:
            The initial state
        """
        # Start with a random historical hurricane at its beginning
        if not self.hurricane_data:
            # Create dummy data if no historical data available
            self.current_state = self._create_dummy_state()
            return self.get_state()
            
        random_index = random.randint(0, len(self.hurricane_data) - 1)
        selected_hurricane = self.hurricane_data[random_index]
        
        self.current_state = {
            "hurricane_id": selected_hurricane.get("id"),
            "name": selected_hurricane.get("name"),
            "basin": selected_hurricane.get("basin"),
            "position": selected_hurricane.get("initial_position"),
            "wind_speed": selected_hurricane.get("initial_wind_speed"),
            "pressure": selected_hurricane.get("initial_pressure"),
            "sea_surface_temp": self.get_nasa_data_for_location(
                selected_hurricane.get("initial_position")
            ),
            "timestamp": selected_hurricane.get("start_time"),
            "actual": selected_hurricane.get("track")  # Full track data for evaluation
        }
        
        self.time_step = 0
        self.history = []
        
        return self.get_state()
    
    def _create_dummy_state(self) -> Dict:
        """Create a dummy state for testing when no data is available."""
        return {
            "hurricane_id": "dummy_hurricane",
            "name": "Test Hurricane",
            "basin": "NA",  # North Atlantic
            "position": {"lat": 25.0, "lon": -75.0},
            "wind_speed": 65.0,
            "pressure": 990.0,
            "sea_surface_temp": {
                "type": "seaSurfaceTemperature",
                "value": 28.5
            },
            "timestamp": pd.Timestamp.now(),
            "actual": []  # No actual track data for dummy
        }
    
    def get_nasa_data_for_location(self, position: Dict) -> Dict:
        """
        Get NASA data for a specific location.
        
        Args:
            position: Dictionary with lat/lon coordinates
            
        Returns:
            NASA data for the location
        """
        if not position or not hasattr(self.nasa_data, 'get_gibs_layers'):
            # Default values if NASA data service not available
            return {
                "type": "seaSurfaceTemperature",
                "value": self.estimate_sea_surface_temp_for_location(position)
            }
        
        # In a complete implementation, this would query NASA data for the coordinates
        try:
            # Estimate SST based on position and climate patterns
            return {
                "type": "seaSurfaceTemperature",
                "value": self.estimate_sea_surface_temp_for_location(position)
            }
        except Exception as e:
            print(f"Error getting NASA data: {e}")
            return {
                "type": "seaSurfaceTemperature",
                "value": 28.0  # Default fallback
            }
    
    def estimate_sea_surface_temp_for_location(self, position: Dict) -> float:
        """
        Estimate sea surface temperature for a location based on position and season.
        This is a simplified model until we integrate real-time NASA data.
        
        Args:
            position: Dictionary with lat/lon coordinates
            
        Returns:
            Estimated sea surface temperature in degrees Celsius
        """
        if not position:
            return 28.0  # Default value
        
        lat = position.get('lat', 0)
        lon = position.get('lon', 0)
        
        # Base temperature based on absolute latitude (warmer near equator)
        latitude_effect = 30 - (abs(lat) * 0.3)
        
        # Longitude effects - customized by ocean basin
        basin_effect = 0
        
        # Simplified basin determination
        basin = 'UNKNOWN'
        if lon > -100 and lon < 0 and lat > 0: 
            basin = 'NA'  # North Atlantic
        elif lon >= -180 and lon < -100 and lat > 0: 
            basin = 'EP'  # Eastern Pacific
        elif lon >= 100 and lon < 180 and lat > 0: 
            basin = 'WP'  # Western Pacific
        elif lon >= 40 and lon < 100 and lat > 0: 
            basin = 'NI'  # North Indian
        elif lon >= 40 and lon < 135 and lat <= 0: 
            basin = 'SI'  # South Indian
        elif lon >= 135 and lon < 180 and lat <= 0: 
            basin = 'SP'  # South Pacific
        
        # Basin-specific effects
        if basin == 'WP':  # Western Pacific tends to be warmer
            basin_effect = 1.5
        elif basin == 'NI':  # North Indian Ocean
            basin_effect = 2.0
        elif basin == 'NA':  # North Atlantic
            basin_effect = 0.0
        else:
            basin_effect = 0.5
        
        # Get current month for seasonal effects
        current_state_timestamp = self.current_state.get('timestamp') if self.current_state else None
        if current_state_timestamp:
            if isinstance(current_state_timestamp, pd.Timestamp):
                current_month = current_state_timestamp.month
            else:
                try:
                    current_month = current_state_timestamp.month
                except:
                    current_month = pd.Timestamp.now().month
        else:
            current_month = pd.Timestamp.now().month
        
        # Seasonal effect (northern and southern hemisphere have opposite seasons)
        seasonal_effect = 0
        if lat > 0:  # Northern hemisphere
            # Warmer in northern summer (Jun-Aug)
            seasonal_effect = math.sin((current_month - 5) * math.pi / 6) * 2
        else:  # Southern hemisphere
            # Warmer in southern summer (Dec-Feb)
            seasonal_effect = math.sin((current_month - 11) * math.pi / 6) * 2
        
        # Calculate final temperature with some randomness
        temperature = latitude_effect + basin_effect + seasonal_effect + (random.random() - 0.5)
        
        # Constrain to realistic values (20-32°C)
        return max(20, min(32, temperature))
    
    def get_state(self) -> Dict:
        """
        Get the current state observation for the agent.
        
        Returns:
            Current state dictionary
        """
        if not self.current_state:
            return {}
            
        return {
            "position": self.current_state.get("position"),
            "wind_speed": self.current_state.get("wind_speed"),
            "pressure": self.current_state.get("pressure"),
            "sea_surface_temp": self.current_state.get("sea_surface_temp"),
            "basin": self.current_state.get("basin"),
            "timestamp": self.current_state.get("timestamp"),
            "time_step": self.time_step
        }
    
    def step(self, action: Dict) -> Dict:
        """
        Take an action in the environment.
        
        Args:
            action: Predicted position and intensity
            
        Returns:
            Dict containing new state, reward, and done flag
        """
        # Record the agent's prediction
        self.history.append({
            "time_step": self.time_step,
            "prediction": action,
            "actual": self.get_actual_for_time_step(self.time_step + 1)
        })
        
        # Update the time step
        self.time_step += 1
        
        # Update the current state to actual values for next time step
        actual_next_state = self.get_actual_for_time_step(self.time_step)
        if actual_next_state:
            self.current_state = {
                **self.current_state,
                "position": actual_next_state.get("position"),
                "wind_speed": actual_next_state.get("wind_speed"),
                "pressure": actual_next_state.get("pressure"),
                "sea_surface_temp": self.get_nasa_data_for_location(
                    actual_next_state.get("position")
                ),
                "timestamp": actual_next_state.get("timestamp")
            }
        
        # Calculate reward based on prediction accuracy
        reward = self.calculate_reward(action, actual_next_state)
        
        # Check if episode is done
        done = (actual_next_state is None or 
                self.time_step >= len(self.current_state.get("actual", [])) - 1)
        
        return {
            "state": self.get_state(),
            "reward": reward,
            "done": done
        }
    
    def get_actual_for_time_step(self, time_step: int) -> Optional[Dict]:
        """
        Get actual hurricane state for a specific time step.
        
        Args:
            time_step: The time step to retrieve
            
        Returns:
            Actual state dictionary or None if not available
        """
        actual_track = self.current_state.get("actual", [])
        if not actual_track or time_step >= len(actual_track):
            return None
        return actual_track[time_step]
    
    def calculate_reward(self, prediction: Dict, actual: Optional[Dict]) -> float:
        """
        Calculate reward based on prediction accuracy.
        
        Args:
            prediction: The agent's prediction
            actual: The actual outcome
            
        Returns:
            Reward value
        """
        if not actual:
            return 0.0
        
        # Calculate positional error (in km)
        position_error = self.calculate_distance(
            prediction.get("position"),
            actual.get("position")
        )
        
        # Calculate intensity error
        intensity_error = abs(prediction.get("wind_speed", 0) - actual.get("wind_speed", 0))
        
        # Calculate pressure error
        pressure_error = abs(prediction.get("pressure", 0) - actual.get("pressure", 0))
        
        # Reward decreases with error
        position_reward = max(0, 100 - position_error)
        intensity_reward = max(0, 50 - intensity_error)
        pressure_reward = max(0, 30 - (pressure_error / 2))
        
        # Weight rewards based on forecast time
        # Position accuracy is more important for short-term forecasts
        # Intensity becomes more important for medium-term forecasts
        forecast_hour = self.time_step * 6  # Assuming 6-hour time steps
        
        if forecast_hour <= 24:
            # Short-term (0-24h): Position is most critical
            position_weight = 0.6
            intensity_weight = 0.3
            pressure_weight = 0.1
        elif forecast_hour <= 72:
            # Medium-term (24-72h): Balance position and intensity
            position_weight = 0.5
            intensity_weight = 0.4
            pressure_weight = 0.1
        else:
            # Long-term (>72h): Intensity trend becomes more important
            position_weight = 0.4
            intensity_weight = 0.5
            pressure_weight = 0.1
        
        return (
            position_reward * position_weight + 
            intensity_reward * intensity_weight + 
            pressure_reward * pressure_weight
        )
    
    def calculate_distance(self, pos1: Optional[Dict], pos2: Optional[Dict]) -> float:
        """
        Calculate distance between two geographic points (in km).
        
        Args:
            pos1: First position with lat/lon
            pos2: Second position with lat/lon
            
        Returns:
            Distance in kilometers
        """
        if not pos1 or not pos2:
            return 1000.0  # Large error if positions missing
        
        # Haversine formula for calculating distance between two points on Earth
        R = 6371.0  # Earth's radius in km
        
        lat1 = pos1.get('lat', 0)
        lon1 = pos1.get('lon', 0)
        lat2 = pos2.get('lat', 0)
        lon2 = pos2.get('lon', 0)
        
        d_lat = self.deg2rad(lat2 - lat1)
        d_lon = self.deg2rad(lon2 - lon1)
        
        a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
             math.cos(self.deg2rad(lat1)) * math.cos(self.deg2rad(lat2)) *
             math.sin(d_lon / 2) * math.sin(d_lon / 2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def deg2rad(self, deg: float) -> float:
        """Convert degrees to radians."""
        return deg * (math.pi / 180)
    
    def evaluate_performance(self) -> Dict:
        """
        Evaluate agent performance over an episode.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate average position and intensity errors
        errors = []
        for record in self.history:
            if not record.get('actual'):
                continue
            
            # Safely calculate position error
            prediction = record.get('prediction', {})
            actual = record.get('actual', {})
            
            pos_error = self.calculate_distance(
                prediction.get('position'),
                actual.get('position')
            )
            
            # Safely calculate intensity error
            intensity_error = abs(
                (prediction.get('wind_speed') or 0) - 
                (actual.get('wind_speed') or 0)
            )
            
            # Safely calculate pressure error
            pressure_error = abs(
                (prediction.get('pressure') or 1000) - 
                (actual.get('pressure') or 1000)
            )
            
            # Calculate error by forecast period
            time_step = record.get('time_step', 0)
            forecast_hour = time_step * 6  # Assuming 6-hour time steps
            
            if forecast_hour <= 24:
                forecast_period = '24h'
            elif forecast_hour <= 48:
                forecast_period = '48h'
            elif forecast_hour <= 72:
                forecast_period = '72h'
            elif forecast_hour <= 96:
                forecast_period = '96h'
            else:
                forecast_period = '120h'
            
            errors.append({
                'pos_error': pos_error if not math.isnan(pos_error) else 0,
                'intensity_error': intensity_error if not math.isnan(intensity_error) else 0,
                'pressure_error': pressure_error if not math.isnan(pressure_error) else 0,
                'forecast_hour': forecast_hour,
                'forecast_period': forecast_period
            })
        
        if not errors:
            return {
                'avg_pos_error': 0,
                'avg_intensity_error': 0,
                'avg_pressure_error': 0,
                'by_forecast_period': {}
            }
        
        # Calculate overall averages
        avg_pos_error = sum(e['pos_error'] for e in errors) / len(errors)
        avg_intensity_error = sum(e['intensity_error'] for e in errors) / len(errors)
        avg_pressure_error = sum(e['pressure_error'] for e in errors) / len(errors)
        
        # Group by forecast period
        by_forecast_period = {}
        periods = ['24h', '48h', '72h', '96h', '120h']
        
        for period in periods:
            period_errors = [e for e in errors if e['forecast_period'] == period]
            
            if not period_errors:
                by_forecast_period[period] = None
                continue
            
            by_forecast_period[period] = {
                'count': len(period_errors),
                'avg_pos_error': sum(e['pos_error'] for e in period_errors) / len(period_errors),
                'avg_intensity_error': sum(e['intensity_error'] for e in period_errors) / len(period_errors),
                'avg_pressure_error': sum(e['pressure_error'] for e in period_errors) / len(period_errors)
            }
        
        return {
            'avg_pos_error': avg_pos_error if not math.isnan(avg_pos_error) else 0,
            'avg_intensity_error': avg_intensity_error if not math.isnan(avg_intensity_error) else 0,
            'avg_pressure_error': avg_pressure_error if not math.isnan(avg_pressure_error) else 0,
            'by_forecast_period': by_forecast_period
        }