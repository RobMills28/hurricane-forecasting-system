"""
Ensemble Coordinator for Hurricane Prediction

This module combines predictions from specialised hurricane prediction agents.
"""

import math
from typing import Dict, List, Optional

from .trajectory_agent import TrajectoryAgent
from .intensity_agent import IntensityAgent
from .basin_specific_agent import BasinSpecificAgent

class EnsembleCoordinator:
    """
    Coordinates multiple specialised agents for hurricane prediction.
    
    This class manages the ensemble of specialised agents and combines
    their predictions based on their expertise and confidence.
    """
    
    def __init__(self, options: Dict = None):
        """Initialise the ensemble coordinator with specialised agents."""
        # Default options
        self.options = {
            "use_basin_specific": False,
            "trajectory_weight": 0.7,     # Increase trajectory weight further
            "intensity_weight": 0.3,      # Keep intensity weight the same
            "basin_weight": 0.0,          # Set to zero
            "dynamic_weighting": False    # Keep dynamic weighting off

        }
        
        # Update with provided options
        if options:
            self.options.update(options)
        
        # Initialise specialised agents
        self.trajectory_agent = TrajectoryAgent()
        self.intensity_agent = IntensityAgent()
        
        # Initialise basin-specific agents
        self.basin_agents = {}
        if self.options["use_basin_specific"]:
            self.basin_agents = {
                "NA": BasinSpecificAgent("NA"),  # North Atlantic
                "WP": BasinSpecificAgent("WP"),  # Western Pacific
                "EP": BasinSpecificAgent("EP"),  # Eastern Pacific
                # Add other basins as needed
            }
    
    def predict(self, state: Dict, history: List[Dict], training: bool = False) -> Dict:
        """
        Make a coordinated prediction using all specialised agents.
        
        Args:
            state: Current hurricane state
            history: List of historical states
            
        Returns:
            Combined prediction from all agents
        """
        # Get predictions from specialised agents
        trajectory_prediction = self.trajectory_agent.predict(state, history, training)
        intensity_prediction = self.intensity_agent.predict(state, history, training)
        
        # Get prediction from basin-specific agent if available
        basin = state.get("basin", "DEFAULT")
        basin_prediction = None
        if basin in self.basin_agents:
            basin_prediction = self.basin_agents[basin].predict(state, history, training)
        
        # Combine predictions with appropriate weighting
        return self.combine_predictions(
            state, 
            trajectory_prediction, 
            intensity_prediction, 
            basin_prediction
        )
    
    def combine_predictions(
        self, 
        state: Dict, 
        trajectory_prediction: Dict, 
        intensity_prediction: Dict, 
        basin_prediction: Optional[Dict] = None
    ) -> Dict:
        """
        Combine predictions from specialised agents with appropriate weighting.
        
        Args:
            state: Current hurricane state
            trajectory_prediction: Prediction from trajectory agent
            intensity_prediction: Prediction from intensity agent
            basin_prediction: Prediction from basin-specific agent (optional)
            
        Returns:
            Combined prediction
        """
        # Extract current position and intensity
        position = state.get("position", {})
        current_lat = position.get("lat", 0)
        current_lon = position.get("lon", 0)
        current_wind = state.get("wind_speed", 0)
        current_pressure = state.get("pressure", 1010)
        
        # Get base weights
        traj_weight = self.options["trajectory_weight"]
        inten_weight = self.options["intensity_weight"]
        basin_weight = self.options["basin_weight"]
        
        # Apply dynamic weighting if enabled
        if self.options["dynamic_weighting"]:
            # Adjust weights based on stage of hurricane
            # Early development: More weight to intensity for rapid intensification
            # Peak intensity: More balance with trajectory important
            # Weakening/dissipation: More weight to trajectory for recurvature
            
            # Simple determination based on current wind speed and latitude
            abs_lat = abs(current_lat)
            
            if current_wind < 60:  # Developing stage
                # Emphasise intensity more for developing storms
                traj_weight *= 0.8
                inten_weight *= 1.2
            elif abs_lat > 30:  # Higher latitude, likely recurving
                # Emphasise trajectory more for recurving storms
                traj_weight *= 1.2
                inten_weight *= 0.8
            elif current_wind > 120:  # Major hurricane stage
                # More balanced at peak intensity, slight emphasis on trajectory
                traj_weight *= 1.1
                inten_weight *= 0.9
        
        # Re-normalise weights if basin prediction is not available
        if basin_prediction is None:
            total = traj_weight + inten_weight
            traj_weight /= total
            inten_weight /= total
            basin_weight = 0
        else:
            # Include basin prediction with confidence factor
            basin_confidence = basin_prediction.get("confidence", 1.0)
            # Adjust basin weight by confidence
            basin_weight *= basin_confidence
            
            # Re-normalise with adjusted basin weight
            total = traj_weight + inten_weight + basin_weight
            if total > 0:
                traj_weight /= total
                inten_weight /= total
                basin_weight /= total
        
        # Extract values from predictions with confidence weighting
        # Position from trajectory prediction
        traj_lat = trajectory_prediction.get("position", {}).get("lat", current_lat)
        traj_lon = trajectory_prediction.get("position", {}).get("lon", current_lon)
        
        # Intensity from intensity prediction
        inten_wind = intensity_prediction.get("wind_speed", current_wind)
        inten_pressure = intensity_prediction.get("pressure", current_pressure)
        
        # Calculate weighted position
        weighted_lat = traj_lat * traj_weight
        weighted_lon = traj_lon * traj_weight
        
        # Calculate weighted intensity
        weighted_wind = inten_wind * inten_weight
        weighted_pressure = inten_pressure * inten_weight
        
        # Add basin prediction if available
        if basin_prediction:
            basin_lat = basin_prediction.get("position", {}).get("lat", current_lat)
            basin_lon = basin_prediction.get("position", {}).get("lon", current_lon)
            basin_wind = basin_prediction.get("wind_speed", current_wind)
            basin_pressure = basin_prediction.get("pressure", current_pressure)
            
            weighted_lat += basin_lat * basin_weight
            weighted_lon += basin_lon * basin_weight
            weighted_wind += basin_wind * basin_weight
            weighted_pressure += basin_pressure * basin_weight
        
        # Calculate uncertainty by combining uncertainties from all agents
        position_uncertainty = {
            "lat": trajectory_prediction.get("uncertainty", {}).get("position", {}).get("lat", 0) * traj_weight,
            "lon": trajectory_prediction.get("uncertainty", {}).get("position", {}).get("lon", 0) * traj_weight
        }
        
        intensity_uncertainty = {
            "wind_speed": intensity_prediction.get("uncertainty", {}).get("wind_speed", 0) * inten_weight,
            "pressure": intensity_prediction.get("uncertainty", {}).get("pressure", 0) * inten_weight
        }
        
        # Add basin uncertainties if available
        if basin_prediction and "uncertainty" in basin_prediction:
            position_uncertainty["lat"] += basin_prediction.get("uncertainty", {}).get("position", {}).get("lat", 0) * basin_weight
            position_uncertainty["lon"] += basin_prediction.get("uncertainty", {}).get("position", {}).get("lon", 0) * basin_weight
            intensity_uncertainty["wind_speed"] += basin_prediction.get("uncertainty", {}).get("wind_speed", 0) * basin_weight
            intensity_uncertainty["pressure"] += basin_prediction.get("uncertainty", {}).get("pressure", 0) * basin_weight
        
        # Validate final values
        weighted_wind = max(30, min(150, weighted_wind))  # Reasonable wind range
        weighted_pressure = max(920, min(1015, weighted_pressure))  # Reasonable pressure range

        # Ensure pressure and wind are physically consistent
        if weighted_wind < 39:  # TD
            weighted_pressure = max(1000, weighted_pressure)
        elif weighted_wind < 74:  # TS
            weighted_pressure = min(1000, max(985, weighted_pressure))
        elif weighted_wind < 96:  # Cat 1
            weighted_pressure = min(985, max(970, weighted_pressure))
        elif weighted_wind < 111:  # Cat 2
            weighted_pressure = min(970, max(955, weighted_pressure))
        elif weighted_wind < 130:  # Cat 3
            weighted_pressure = min(955, max(935, weighted_pressure))
        elif weighted_wind < 157:  # Cat 4
            weighted_pressure = min(935, max(915, weighted_pressure))
        else:  # Cat 5
            weighted_pressure = min(915, weighted_pressure)

        # Combine all into final prediction
        return {
            "position": {"lat": weighted_lat, "lon": weighted_lon},
            "wind_speed": weighted_wind,
            "pressure": weighted_pressure,
            "uncertainty": {
                "position": position_uncertainty,
                "wind_speed": intensity_uncertainty["wind_speed"],
                "pressure": intensity_uncertainty["pressure"]
            },
            "agent_weights": {
                "trajectory": traj_weight,
                "intensity": inten_weight,
                "basin": basin_weight
            }
        }
    
    def train(self, environment, episodes=100):
        """
        Train all specialised agents in the ensemble.
        
        Args:
            environment: The hurricane environment for training
            episodes: Number of episodes to train
        """
        # Train trajectory agent
        print("Training trajectory agent...")
        self.trajectory_agent.train(environment, episodes)
        
        # Train intensity agent
        print("Training intensity agent...")
        self.intensity_agent.train(environment, episodes)
        
        # Train basin-specific agents
        if self.options["use_basin_specific"]:
            for basin, agent in self.basin_agents.items():
                print(f"Training basin-specific agent for {basin}...")
                agent.train(environment, episodes)