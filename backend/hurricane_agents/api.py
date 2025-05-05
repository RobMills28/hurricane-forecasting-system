"""
API server for hurricane prediction models

This module provides an interface between the JavaScript frontend and Python backend.
"""
import torch
from .agent import HurricanePredictionAgent, ReplayBuffer
import pandas as pd
from .utils import determine_basin, haversine_distance
import asyncio
import json
import os
import logging
import random
import math
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

# Import our hurricane agents components
from .environment import HurricaneEnvironment
from .agent import HurricanePredictionAgent, ReplayBuffer
from .data import fetch_historical_hurricane_data, preprocess_data_for_training
from .utils import get_hurricane_category, safe_get

# Openmeteo imports
from .openmeteo_connector import (
    fetch_open_meteo_data,
    get_active_hurricanes_by_region
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hurricane_agents.api")

# Create FastAPI app
app = FastAPI(title="Hurricane Prediction API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store trained agents in memory (in production, you'd use a persistent store)
trained_agents = {}
environments = {}
training_tasks = {}

# Create and initialize a default hurricane prediction agent on startup
@app.on_event("startup")
async def startup_event():
    # Create default agent if it doesn't exist
    if "default-hurricane-agent" not in trained_agents:
        logger.info("Creating default hurricane prediction agent")
        try:
            # Initialize a basic agent
            agent = HurricanePredictionAgent()
            
            # Store the agent
            trained_agents["default-hurricane-agent"] = agent
            
            logger.info("Default hurricane prediction agent created")
        except Exception as e:
            logger.error(f"Error creating default agent: {e}")
            import traceback
            logger.error(traceback.format_exc())

# Model definitions for API requests and responses
class TrainingOptions(BaseModel):
    agent_id: str
    episodes: int = 100
    basin: Optional[str] = None
    use_basin_models: bool = True
    use_dynamic_weights: bool = True
    ensemble_size: int = 5
    min_category: int = 0

class PredictionRequest(BaseModel):
    agent_id: str
    current_state: Dict[str, Any]
    history: List[Dict[str, Any]] = []

class TrainingStatus(BaseModel):
    agent_id: str
    status: str
    progress: float = 0
    metrics: Optional[Dict[str, Any]] = None

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

# Update the StormDataRequest model to be more flexible
class StormDataRequest(BaseModel):
    coordinates: Optional[Union[List[float], List[Any], Dict[str, float]]] = None
    basin: Optional[str] = None
    hurricane_id: Optional[str] = None
    name: Optional[str] = None
    category: Optional[Any] = None  # Accept any type for category
    data_source: Optional[str] = None
    
    class Config:
        # Make validation more flexible
        extra = "allow"  # Allow extra fields in the request
        arbitrary_types_allowed = True

# Background task to train an agent
async def train_agent_task(agent_id: str, options: Dict[str, Any]):
    """Background task to train an agent using reinforcement learning."""
    try:
        logger.info(f"Starting training for agent {agent_id}")
        training_tasks[agent_id]["status"] = "loading_data"
        
        # Fetch and preprocess data
        data_options = {
            "min_category": options.get("min_category", 0),
            "basin": options.get("basin"),
        }
        historical_data = await fetch_historical_hurricane_data(data_options)
        processed_data = preprocess_data_for_training(historical_data)
        
        # Create environment
        environment = HurricaneEnvironment()
        await environment.initialize(processed_data, {})
        environments[agent_id] = environment
        
        # Create agent
        agent_options = {
            "use_basin_models": options.get("use_basin_models", True),
            "use_dynamic_weights": options.get("use_dynamic_weights", True),
            "ensemble_size": options.get("ensemble_size", 5),
            "state_dim": 12,  # Adjust based on state representation
            "action_dim": 15   # 5 lat dirs x 3 lon dirs x 1 intensity
        }
        agent = HurricanePredictionAgent(agent_options)
        
        # Train agent using reinforcement learning
        training_tasks[agent_id]["status"] = "training"
        episodes = options.get("episodes", 100)
        
        # Update progress as training proceeds
        for episode in range(1, episodes + 1):
            # Reset for this episode
            state = environment.reset()
            
            done = False
            total_reward = 0
            
            # Episode loop
            while not done:
                # Make prediction based on current state
                action = agent.predict(state, [], training=True)
                
                # Take action and observe result
                result = environment.step(action)
                next_state = result.get("state", {})
                reward = result.get("reward", 0)
                done = result.get("done", False)
                
                # Update agent using reinforcement learning
                agent.update(state, action, reward, next_state, done)
                
                # Track total reward
                total_reward += reward
                
                # Update state
                state = next_state
            
            # Evaluate performance
            episode_performance = environment.evaluate_performance()
            
            # Update training metrics
            episode_performance["episode_reward"] = total_reward
            
            # Update progress
            training_tasks[agent_id]["progress"] = episode / episodes
            training_tasks[agent_id]["metrics"] = episode_performance
            
            # Log progress every 10 episodes
            if episode % 10 == 0 or episode == 1 or episode == episodes:
                logger.info(
                    f"Episode {episode}/{episodes} - "
                    f"Reward: {total_reward:.2f}, "
                    f"Avg Position Error: {episode_performance.get('avg_pos_error', 0):.2f} km, "
                    f"Avg Intensity Error: {episode_performance.get('avg_intensity_error', 0):.2f} mph"
                )
            
            # Add a small delay to prevent blocking
            await asyncio.sleep(0.01)
        
        # Store trained agent
        trained_agents[agent_id] = agent
        training_tasks[agent_id]["status"] = "completed"
        logger.info(f"Training completed for agent {agent_id}")
        
    except Exception as e:
        logger.error(f"Error training agent {agent_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        training_tasks[agent_id]["status"] = "failed"
        training_tasks[agent_id]["error"] = str(e)

# Ocean basin identifiers mapped to coordinates
BASIN_COORDINATES = {
    'WP': {'name': 'Western Pacific', 'hotspots': [
        {'lat': 15.0, 'lon': 130.0, 'name': 'Philippine Sea'},
        {'lat': 20.0, 'lon': 135.0, 'name': 'Western Pacific'},
        {'lat': 10.0, 'lon': 145.0, 'name': 'Micronesia Region'}
    ]},
    'EP': {'name': 'Eastern Pacific', 'hotspots': [
        {'lat': 15.0, 'lon': -105.0, 'name': 'Eastern Pacific'},
        {'lat': 12.0, 'lon': -120.0, 'name': 'Central Pacific'}
    ]},
    'NA': {'name': 'North Atlantic', 'hotspots': [
        {'lat': 25.0, 'lon': -75.0, 'name': 'Western Atlantic'},
        {'lat': 15.0, 'lon': -55.0, 'name': 'Central Atlantic'},
        {'lat': 20.0, 'lon': -85.0, 'name': 'Caribbean Sea'}
    ]},
    'NI': {'name': 'North Indian', 'hotspots': [
        {'lat': 15.0, 'lon': 85.0, 'name': 'Bay of Bengal'},
        {'lat': 15.0, 'lon': 65.0, 'name': 'Arabian Sea'}
    ]},
    'SI': {'name': 'South Indian', 'hotspots': [
        {'lat': -15.0, 'lon': 70.0, 'name': 'Madagascar Basin'},
        {'lat': -15.0, 'lon': 90.0, 'name': 'Eastern Indian Ocean'}
    ]},
    'SP': {'name': 'South Pacific', 'hotspots': [
        {'lat': -15.0, 'lon': 170.0, 'name': 'Fiji Basin'},
        {'lat': -15.0, 'lon': 145.0, 'name': 'Coral Sea'}
    ]}
}

@app.post("/train")
async def train_agent(options: TrainingOptions, background_tasks: BackgroundTasks):
    """
    Start training a hurricane prediction agent.
    
    This is an asynchronous operation - the training happens in the background.
    Check the status using the /training_status endpoint.
    """
    agent_id = options.agent_id
    
    # Check if this agent is already training
    if agent_id in training_tasks and training_tasks[agent_id]["status"] in ["loading_data", "training"]:
        return {
            "message": f"Agent {agent_id} is already training",
            "agent_id": agent_id,
            "status": training_tasks[agent_id]["status"],
            "progress": training_tasks[agent_id]["progress"]
        }
    
    # Initialize training task
    training_tasks[agent_id] = {
        "status": "starting",
        "progress": 0,
        "metrics": None
    }
    
    # Start background training task
    background_tasks.add_task(train_agent_task, agent_id, options.dict())
    
    return {
        "message": f"Started training agent {agent_id}",
        "agent_id": agent_id,
        "status": "starting",
    }

@app.get("/training_status/{agent_id}")
async def get_training_status(agent_id: str):
    """Get the status of an agent's training."""
    if agent_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    status = training_tasks[agent_id]["status"]
    progress = training_tasks[agent_id]["progress"]
    metrics = training_tasks[agent_id].get("metrics")
    error = training_tasks[agent_id].get("error")
    
    response = {
        "agent_id": agent_id,
        "status": status,
        "progress": progress,
    }
    
    if metrics:
        response["metrics"] = metrics
    
    if error:
        response["error"] = error
    
    return response

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Make a prediction using a trained agent.
    
    The agent must have been previously trained.
    """
    agent_id = request.agent_id
    
    if agent_id not in trained_agents:
        raise HTTPException(
            status_code=404, 
            detail=f"Agent {agent_id} not found or not trained yet"
        )
    
    agent = trained_agents[agent_id]
    
    # Convert the request state and history to the format expected by the agent
    state = request.current_state
    history = request.history
    
    # Make prediction (not in training mode)
    prediction = agent.predict(state, history, training=False)
    
    # Add hurricane category based on wind speed
    if "wind_speed" in prediction:
        prediction["category"] = get_hurricane_category(prediction["wind_speed"])
    
    return prediction

@app.post("/storm_data")
async def get_storm_data(request: StormDataRequest):
    """
    Get comprehensive storm data including observations, forecasts, and risk analysis.
    This centralizes data processing in the Python backend.
    """
    try:
        # Debug logging to see what's being received
        logger.info(f"Received request with coordinates: {request.coordinates}")
        logger.info(f"Full request: {request}")
        
        # Handle coordinates in various formats
        coordinates = request.coordinates
        if not coordinates:
            raise HTTPException(status_code=400, detail="Coordinates are required")
            
        # Extract lat/lon from different possible formats
        try:
            if isinstance(coordinates, list):
                if len(coordinates) >= 2:
                    # [lon, lat] format
                    lon, lat = float(coordinates[0]), float(coordinates[1])
                else:
                    raise ValueError("List coordinates must have at least 2 values")
            elif isinstance(coordinates, dict):
                # {lat: x, lon: y} format
                lat = float(coordinates.get('lat', 0))
                lon = float(coordinates.get('lon', 0))
            else:
                raise ValueError(f"Unsupported coordinates format: {type(coordinates)}")
        except (ValueError, TypeError) as e:
            logger.error(f"Coordinate parsing error: {e} for coordinates: {coordinates}")
            raise HTTPException(status_code=422, detail=f"Unable to parse coordinates: {str(e)}")
        
        # Log what we've parsed
        logger.info(f"Parsed coordinates: lat={lat}, lon={lon}")
        
        # Determine the data source based on region or request
        # For US regions, use NOAA. For others, use OpenMeteo
        data_source = request.data_source
        basin = request.basin or determine_basin(lon, lat)
        
        # Determine if this is a US-based storm
        is_us_storm = is_in_us_region(lat, lon)
        
        # If data source wasn't specified, choose based on region
        if not data_source:
            data_source = 'noaa' if is_us_storm else 'openmeteo'
        
        # Get observations from the appropriate source
        if data_source.lower() == 'openmeteo' or (not is_us_storm and data_source.lower() != 'noaa'):
            # Use OpenMeteo for non-US regions
            logger.info(f"Using OpenMeteo data for non-US coordinates: {lat}, {lon}")
            try:
                storm_data = await fetch_open_meteo_data(lat, lon, basin)
                
                # If it failed, still try NOAA as fallback
                if 'error' in storm_data:
                    logger.warning(f"OpenMeteo failed, falling back to NOAA: {storm_data.get('message')}")
                    observations = await get_hurricane_observations(lat, lon)
                else:
                    # Use OpenMeteo data
                    observations = storm_data.get('observations')
                    forecast_data = storm_data.get('forecast', [])
                    risk_level = storm_data.get('riskLevel', 'moderate')
                    
                    # If we have forecast data from OpenMeteo, use it directly
                    if forecast_data:
                        logger.info(f"Using OpenMeteo forecast data with {len(forecast_data)} points")
                        return {
                            "observations": observations,
                            "forecast": forecast_data,
                            "riskLevel": risk_level,
                            "satelliteImagery": get_satellite_imagery(lat, lon),
                            "historicalData": generate_historical_data()
                        }
            except Exception as e:
                logger.error(f"Error getting OpenMeteo data: {e}")
                # Fall back to NOAA
                observations = await get_hurricane_observations(lat, lon)
        else:
            # Use NOAA for US regions
            logger.info(f"Using NOAA data for US coordinates: {lat}, {lon}")
            observations = await get_hurricane_observations(lat, lon)
        
        # Create state object for prediction
        current_state = {
            "position": {"lat": lat, "lon": lon},
            "wind_speed": observations.get("windSpeed", 75),
            "pressure": observations.get("barometricPressure", 990),
            "basin": basin,
            "sea_surface_temp": {"value": observations.get("temperature", 28.5)}
        }
        logger.info(f"Created state object: {current_state}")
        
        # Get agent prediction with enhanced error handling
        try:
            agent_id = "default-hurricane-agent"
            agent = trained_agents.get(agent_id)
            
            # If no trained agent exists, create a default one
            if not agent:
                logger.info("Creating new default agent")
                agent = HurricanePredictionAgent()
                trained_agents[agent_id] = agent
            
            logger.info("Calling agent.predict")
            prediction = agent.predict(current_state, [])
            logger.info(f"Agent prediction: {prediction}")
        except Exception as e:
            logger.error(f"Error in agent prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Use a default prediction if agent fails
            prediction = {
                "position": {"lat": lat, "lon": lon},
                "wind_speed": observations.get("windSpeed", 75),
                "pressure": observations.get("barometricPressure", 990)
            }
            logger.info(f"Using default prediction: {prediction}")
        
        # Generate forecast from the prediction
        forecast = generate_forecast_points(prediction, current_state)
        
        # Calculate risk level based on the forecast
        risk_level = calculate_risk_level(prediction, observations)
        
        # Return all data needed by the frontend
        return {
            "observations": observations,
            "forecast": forecast,
            "riskLevel": risk_level,
            "satelliteImagery": get_satellite_imagery(lat, lon),
            "historicalData": generate_historical_data()
        }
            
    except Exception as e:
        logger.error(f"Error in get_storm_data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trained_agents")
async def list_trained_agents():
    """List all trained agents."""
    return {
        "agents": [
            {
                "agent_id": agent_id,
                "status": "trained" if agent_id in trained_agents else training_tasks.get(agent_id, {}).get("status", "unknown")
            }
            for agent_id in set(list(trained_agents.keys()) + list(training_tasks.keys()))
        ]
    }

@app.delete("/agent/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete a trained agent."""
    if agent_id in trained_agents:
        del trained_agents[agent_id]
    
    if agent_id in environments:
        del environments[agent_id]
    
    if agent_id in training_tasks:
        del training_tasks[agent_id]
    
    return {"message": f"Agent {agent_id} deleted"}

# For testing: simple health check endpoint
@app.get("/")
async def root():
    """API health check."""
    return {"status": "healthy", "message": "Hurricane Prediction API is running"}

@app.get("/potential_storm_areas")
async def get_potential_storm_areas():
    """
    Get potential storm formation areas based on environmental conditions.
    This uses the RL agent to predict areas where new storms might form.
    """
    try:
        logger.info("Fetching potential storm formation areas using RL models")
        
        # Get latest global environmental data
        environmental_data = await get_global_environmental_data()
        
        # Use the agent system to predict formations using historical patterns and RL
        potential_areas = generate_storm_formation_predictions(environmental_data)
        
        # Log predictions
        logger.info(f"Generated {len(potential_areas)} potential storm formation areas")
        for area in potential_areas:
            logger.info(f"Potential formation: {area['basin']} at {area['position']} with {area['probability']:.2f} probability")
        
        return {"potential_areas": potential_areas}
    except Exception as e:
        logger.error(f"Error in get_potential_storm_areas: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return simulated data for testing if the actual prediction fails
        return {
            "potential_areas": generate_simulated_formation_areas()
        }

@app.get("/global_storms")
async def get_global_storms(region: str = "GLOBAL"):
    """
    Get active hurricanes and severe storms globally or by region
    
    Args:
        region: Optional ocean basin code (NA, EP, WP, NI, SI, SP)
        
    Returns:
        List of active storms
    """
    try:
        logger.info(f"Fetching global storms for region: {region}")
        
        # Get US storms from NOAA
        us_storms = []
        try:
            us_storms = await get_active_hurricanes()
            logger.info(f"Found {len(us_storms)} US storms from NOAA")
        except Exception as e:
            logger.error(f"Error fetching US storms: {e}")
        
        # Get global storms from OpenMeteo
        global_storms = []
        try:
            global_storms = await get_active_hurricanes_by_region(region)
            logger.info(f"Found {len(global_storms)} global storms from OpenMeteo")
        except Exception as e:
            logger.error(f"Error fetching global storms: {e}")
        
        # Combine storms, removing duplicates based on coordinates proximity
        all_storms = us_storms + global_storms
        unique_storms = []
        processed_coords = set()
        
        # Sort by category strength (highest first) and prefer NOAA sources over OpenMeteo
        sorted_storms = sorted(
            all_storms, 
            key=lambda s: (
                # Priority by category (higher first)
                -int(s.get('category') if s.get('category') not in ['TD', 'TS'] else 0),
                # Then prefer NOAA sources over OpenMeteo
                0 if s.get('dataSource') == 'NOAA' else 1
            )
        )
        
        # Filter based on coordinate proximity (2-degree grid cells)
        for storm in sorted_storms:
            if not storm.get('coordinates'):
                continue
                
            # Get grid cell for coordinates
            coords = storm.get('coordinates')
            if len(coords) < 2:
                continue
                
            lon, lat = coords[0], coords[1]
            grid_key = f"{int(lat/2)*2}_{int(lon/2)*2}"
            
            if grid_key not in processed_coords:
                unique_storms.append(storm)
                processed_coords.add(grid_key)
        
        return {"storms": unique_storms}
    except Exception as e:
        logger.error(f"Error in get_global_storms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions for the storm_data endpoint
async def get_storm_observations(lat, lon, data_source, basin):
    """Get observations from appropriate weather data source."""
    # For now, just return simulated data
    # In a complete implementation, this would call external APIs
    return {
        "timestamp": "2025-04-15T00:00:00Z",
        "temperature": 28.5,
        "windSpeed": 75.0 + (random.random() * 30),
        "windDirection": 120.0 + (random.random() * 60),
        "barometricPressure": 985.0 + (random.random() * 15),
        "relativeHumidity": 85.0 + (random.random() * 10)
    }

def generate_forecast_points(prediction, current_state):
    """Generate forecast points for the UI from agent prediction."""
    forecast = []
    
    # Generate points for the next 5 days (120 hours)
    for hour in range(0, 121, 6):
        day = hour // 24 + 1
        
        # Uncertainty increases with time
        uncertainty_factor = min(0.5, 0.1 + ((hour / 24) * 0.1))
        
        # Base forecast on the prediction with increasing uncertainty
        wind_speed = prediction.get("wind_speed", current_state.get("wind_speed", 75))
        pressure = prediction.get("pressure", current_state.get("pressure", 990))
        
        # Calculate uncertainty ranges
        wind_low = wind_speed * (1 - uncertainty_factor)
        wind_high = wind_speed * (1 + uncertainty_factor)
        
        # Get position with interpolation over time
        predicted_position = prediction.get("position", {})
        current_position = current_state.get("position", {})
        
        # Calculate position at this forecast hour
        if hour == 0:
            position = current_position
        else:
            # Simple linear interpolation - in reality would be more complex
            factor = min(1.0, hour / 24)  # Cap at 1.0 (24 hours)
            lat = current_position.get("lat", 0) + (
                (predicted_position.get("lat", 0) - current_position.get("lat", 0)) * factor
            )
            lon = current_position.get("lon", 0) + (
                (predicted_position.get("lon", 0) - current_position.get("lon", 0)) * factor
            )
            position = {"lat": lat, "lon": lon}
        
        forecast.append({
            "hour": hour,
            "day": day,
            "windSpeed": wind_speed,
            "pressure": pressure,
            "category": get_hurricane_category(wind_speed),
            "windLow": wind_low,
            "windHigh": wind_high,
            "position": position,
            "confidence": max(20, round(100 - (hour * 0.6)))
        })
    
    return forecast

# Helper function to determine if coordinates are in US region
def is_in_us_region(lat: float, lon: float) -> bool:
    """Check if coordinates are in US regions (including territories)"""
    # Continental US
    if 24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0:
        return True
    
    # Alaska
    if 51.0 <= lat <= 72.0 and -170.0 <= lon <= -130.0:
        return True
    
    # Hawaii
    if 18.0 <= lat <= 23.0 and -161.0 <= lon <= -154.0:
        return True
    
    # Puerto Rico and US Virgin Islands
    if 17.5 <= lat <= 18.5 and -67.5 <= lon <= -64.5:
        return True
    
    # Guam and Northern Mariana Islands
    if 13.0 <= lat <= 21.0 and 144.0 <= lon <= 146.0:
        return True
    
    return False

def calculate_risk_level(prediction, observations):
    """Calculate risk level based on prediction and observations."""
    wind_speed = prediction.get("wind_speed", 0)
    
    if wind_speed > 130:
        return "extreme"
    elif wind_speed > 110:
        return "high"
    elif wind_speed > 74:
        return "moderate"
    else:
        return "low"

def get_satellite_imagery(lat, lon):
    """Get satellite imagery for the specified location."""
    # Simulated satellite imagery data
    return {
        "url": None,  # Would be a real image URL in production
        "date": "2025-04-15",
        "resolution": "250m"
    }

def generate_historical_data():
    """Generate historical data for the storm."""
    # For now, return simulated data
    history = []
    for i in range(-168, 0, 6):
        history.append({
            "hour": i,
            "windSpeed": 50 + (random.random() * 30),
            "pressure": 995 - (random.random() * 15)
        })
    return history

async def get_global_environmental_data():
    """
    Fetch global environmental data relevant for storm formation prediction.
    """
    try:
        # In a full implementation, this would call external APIs or services
        # For now, return simulated environmental data
        return {
            "sea_surface_temps": {
                "atlantic": 28.5,
                "eastern_pacific": 29.2,
                "western_pacific": 30.1,
                "indian_ocean": 28.9
            },
            "wind_shear": {
                "atlantic": "low",
                "eastern_pacific": "moderate",
                "western_pacific": "low",
                "indian_ocean": "high"
            },
            "pressure_systems": [
                {"position": [15.0, -40.0], "pressure": 1005, "type": "low"},
                {"position": [12.0, 145.0], "pressure": 1008, "type": "low"},
                {"position": [18.0, -110.0], "pressure": 1007, "type": "low"}
            ],
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching environmental data: {e}")
        return {}

def predict_potential_storm_areas(environmental_data):
    """
    Use the agent system to predict areas where storms might form.
    """
    try:
        logger.info("Using RL agent to predict potential storm formations")
        
        # Use the new implementation that properly uses the agent
        return generate_storm_formation_predictions(environmental_data)
    except Exception as e:
        logger.error(f"Error predicting storm areas: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return simulated data as fallback
        return generate_simulated_formation_areas()

def generate_simulated_formation_areas():
    """
    Generate simulated potential storm formation areas for testing.
    This is used as a fallback when the model-based prediction fails.
    """
    # Common formation regions during hurricane season
    formation_regions = [
        # Atlantic Basin
        {"position": [12.0, -45.0], "basin": "NA"},  # Central Atlantic
        {"position": [15.0, -60.0], "basin": "NA"},  # Eastern Caribbean
        {"position": [23.0, -85.0], "basin": "NA"},  # Western Caribbean/Gulf of Mexico
        
        # Eastern Pacific
        {"position": [13.0, -105.0], "basin": "EP"},  # Off Mexico
        {"position": [11.0, -120.0], "basin": "EP"},  # Eastern Pacific
        
        # Western Pacific
        {"position": [15.0, 130.0], "basin": "WP"},  # Philippine Sea
        {"position": [12.0, 145.0], "basin": "WP"},  # Micronesia
        {"position": [18.0, 160.0], "basin": "WP"},  # Western Pacific
        
        # Indian Ocean
        {"position": [12.0, 65.0], "basin": "NI"},  # Arabian Sea
        {"position": [15.0, 85.0], "basin": "NI"},  # Bay of Bengal
        {"position": [-12.0, 60.0], "basin": "SI"},  # South Indian
        {"position": [-15.0, 90.0], "basin": "SI"},  # SE Indian
        
        # South Pacific
        {"position": [-15.0, 170.0], "basin": "SP"},  # South Pacific
        {"position": [-18.0, 155.0], "basin": "SP"},  # Coral Sea
    ]
    
    # Filter regions based on current season
    current_month = datetime.now().month
    active_regions = []
    for region in formation_regions:
        basin = region["basin"]
        # Northern hemisphere hurricane season: June-November
        if basin in ["NA", "EP", "WP", "NI"] and 6 <= current_month <= 11:
            active_regions.append(region)
        # Southern hemisphere hurricane season: November-April
        elif basin in ["SI", "SP"] and (current_month >= 11 or current_month <= 4):
            active_regions.append(region)
    
    # If no active regions based on season, return some anyway for testing
    if not active_regions:
        active_regions = formation_regions[:5]
    
    # Generate potential formation areas with simulated probabilities
    potential_areas = []
    for i, region in enumerate(active_regions):
        # Random probability, but higher for main formation regions
        probability = 0.3 + (random.random() * 0.5)
        
        # Add variation to position to avoid straight lines
        lat_jitter = (random.random() * 3 - 1.5)  # ±1.5 degrees
        lon_jitter = (random.random() * 3 - 1.5)  # ±1.5 degrees
        
        # Get base position
        position = region["position"]
        basin = region["basin"]
        
        # Random intensity based on probability
        intensity = "TS" if probability > 0.6 else "TD"
        
        potential_areas.append({
            "id": f"pot_{basin}_{i}",
            "position": [position[0] + lat_jitter, position[1] + lon_jitter],
            "probability": probability,
            "basin": basin,
            "intensity": intensity
        })
    
    # Sort by probability and return top areas
    potential_areas.sort(key=lambda x: x["probability"], reverse=True)
    return potential_areas[:7]  # Return top 7 most likely formation areas

def get_basin_sst(basin, sst_data):
    """Get sea surface temperature for a specific basin."""
    basin_map = {
        "NA": "atlantic",
        "EP": "eastern_pacific",
        "WP": "western_pacific",
        "NI": "indian_ocean",
        "SI": "indian_ocean",
        "SP": "western_pacific"
    }
    
    basin_key = basin_map.get(basin, "atlantic")
    return sst_data.get(basin_key, 28.0)

def get_basin_wind_shear(basin, wind_shear):
    """Get wind shear conditions for a specific basin."""
    basin_map = {
        "NA": "atlantic",
        "EP": "eastern_pacific",
        "WP": "western_pacific",
        "NI": "indian_ocean",
        "SI": "indian_ocean",
        "SP": "western_pacific"
    }
    
    basin_key = basin_map.get(basin, "atlantic")
    shear = wind_shear.get(basin_key, "moderate")
    
    # Convert text to numerical value
    shear_values = {"low": 0.9, "moderate": 0.6, "high": 0.3}
    return shear_values.get(shear, 0.6)

def calculate_formation_probability(state, prediction, environmental_data):
    """
    Calculate probability of storm formation using agent predictions and environmental factors.
    """
    position = state.get("position", {})
    lat = position.get("lat", 0)
    lon = position.get("lon", 0)
    basin = state.get("basin", "UNKNOWN")
    sst = state.get("sea_surface_temp", {}).get("value", 28)
    
    # Extract data from prediction
    predicted_wind = prediction.get("wind_speed", 0)
    predicted_pressure = prediction.get("pressure", 1010)
    uncertainty = prediction.get("uncertainty", {})
    
    # Check if location already has low pressure system nearby
    nearby_pressure_system = False
    for system in environmental_data.get("pressure_systems", []):
        sys_pos = system.get("position", [0, 0])
        if haversine_distance(
            {"lat": lat, "lon": lon},
            {"lat": sys_pos[0], "lon": sys_pos[1]}
        ) < 500:  # Within 500km
            nearby_pressure_system = True
            break
    
    # Environmental factors
    # 1. SST factor - higher SST is more favorable (above 26°C)
    sst_factor = max(0, min(1, (sst - 26) / 4)) if sst > 26 else 0
    
    # 2. Pressure prediction - lower predicted pressure is more favorable
    pressure_factor = max(0, min(1, (1013 - predicted_pressure) / 20))
    
    # 3. Wind shear factor from state
    wind_shear_factor = state.get("wind_shear", 0.6)  # Higher is better (less shear)
    
    # 4. Latitude factor - formations typically occur between 5-20° latitude
    abs_lat = abs(lat)
    lat_factor = 0
    if 5 <= abs_lat <= 20:
        lat_factor = 1.0
    elif abs_lat < 5:
        lat_factor = abs_lat / 5
    elif 20 < abs_lat <= 30:
        lat_factor = max(0, (30 - abs_lat) / 10)
    
    # 5. Nearby pressure system bonus
    pressure_system_bonus = 0.2 if nearby_pressure_system else 0
    
    # 6. Season factor
    current_month = datetime.now().month
    season_factor = get_seasonal_factor(basin, current_month)
    
    # 7. Basin-specific historical formation frequency
    basin_factor = get_basin_formation_factor(basin)
    
    # 8. Agent confidence factor (inverse of uncertainty)
    confidence_factor = 0.5
    if uncertainty:
        pos_uncertainty = uncertainty.get("position", {})
        wind_uncertainty = uncertainty.get("wind_speed", 0)
        
        # Lower uncertainty = higher confidence
        position_spread = math.sqrt(
            pos_uncertainty.get("lat", 0.5)**2 + 
            pos_uncertainty.get("lon", 0.5)**2
        )
        confidence_factor = max(0, min(1, 1 - position_spread / 2))
    
    # Weighted combination of factors
    combined = (
        sst_factor * 0.25 +
        pressure_factor * 0.20 +
        wind_shear_factor * 0.15 +
        lat_factor * 0.10 +
        season_factor * 0.10 +
        basin_factor * 0.05 +
        confidence_factor * 0.10 +
        pressure_system_bonus * 0.05
    )
    
    # Apply some randomness for natural variation (reduced to ±5%)
    random_factor = 1.0 + (random.random() * 0.1 - 0.05)  
    
    # Return final probability
    return min(0.95, max(0.1, combined * random_factor))

def generate_storm_formation_predictions(environmental_data):
    """
    Use RL agent to predict potential storm formations based on environmental data.
    """
    # Get global environmental data including SST, pressure systems, etc.
    sst_data = environmental_data.get("sea_surface_temps", {})
    wind_shear = environmental_data.get("wind_shear", {})
    pressure_systems = environmental_data.get("pressure_systems", [])
    
    potential_areas = []
    
    # Use the RL model to predict formations - similar to hurricane tracking but for genesis
    if "default-hurricane-agent" in trained_agents:
        agent = trained_agents["default-hurricane-agent"]
        
        # For each potential formation region, use agent to predict probability
        for basin, region_data in BASIN_COORDINATES.items():
            for hotspot in region_data['hotspots']:
                # Create state for prediction at this location
                state = {
                    "position": {"lat": hotspot["lat"], "lon": hotspot["lon"]},
                    "basin": basin,
                    "sea_surface_temp": {"value": get_basin_sst(basin, sst_data)},
                    "wind_shear": get_basin_wind_shear(basin, wind_shear)
                }
                
                # Use agent to predict formation probability
                prediction = agent.predict(state, [], training=False)
                
                # Calculate probability based on environmental factors and prediction
                formation_probability = calculate_formation_probability(
                    state, prediction, environmental_data
                )
                
                # Only add if probability exceeds threshold
                if formation_probability > 0.2:
                    # Apply jitter to prevent straight-line formations
                    lat_jitter = (random.random() * 3 - 1.5)  # ±1.5 degree variation
                    lon_jitter = (random.random() * 3 - 1.5)  # ±1.5 degree variation
                    
                    potential_areas.append({
                        "id": f"formation_{basin}_{len(potential_areas)}",
                        "position": [hotspot["lat"] + lat_jitter, hotspot["lon"] + lon_jitter],
                        "probability": formation_probability,
                        "basin": basin,
                        "intensity": predict_formation_intensity(state, prediction)
                    })
    
    # If no predictions, fall back to simulated data
    if not potential_areas:
        return generate_simulated_formation_areas()
    
    # Ensure global coverage by adding at least one formation per major basin
    basins_with_formations = {area["basin"] for area in potential_areas}
    major_basins = ["NA", "EP", "WP", "NI", "SI", "SP"]
    
    # For any basin without a formation, add one with lower probability
    for basin in major_basins:
        if basin not in basins_with_formations and basin in BASIN_COORDINATES:
            # Select a random hotspot for this basin
            region_data = BASIN_COORDINATES[basin]
            if 'hotspots' in region_data and region_data['hotspots']:
                hotspot = random.choice(region_data['hotspots'])
                
                # Add jitter to position
                lat_jitter = (random.random() * 3 - 1.5)
                lon_jitter = (random.random() * 3 - 1.5)
                
                # Add a formation with moderate probability
                potential_areas.append({
                    "id": f"formation_{basin}_global",
                    "position": [hotspot["lat"] + lat_jitter, hotspot["lon"] + lon_jitter],
                    "probability": 0.3 + (random.random() * 0.2),  # 0.3-0.5 probability
                    "basin": basin,
                    "intensity": "TD"  # Default to tropical depression
                })
        
    # Filter to avoid too many points - prioritize high probability
    potential_areas.sort(key=lambda x: x["probability"], reverse=True)
    return potential_areas[:15]  # Return top 15 most likely formation areas (increased from 10)

# Create an alias for backward compatibility
async def get_hurricane_observations(lat, lon, data_source=None, basin=None):
    """Alias for get_storm_observations to maintain backward compatibility"""
    return await get_storm_observations(lat, lon, data_source, basin)

def get_seasonal_factor(basin, current_month):
    """Get seasonal factor for storm formation probability based on basin and month."""
    # Northern hemisphere basins peak in August-September
    if basin in ['NA', 'EP', 'WP', 'NI']:
        # Peak: June-November, maximum in August-September
        if 6 <= current_month <= 11:
            # Higher values during peak months (Aug-Sep)
            if 8 <= current_month <= 9:
                return 0.9
            # Moderate values during normal season months
            return 0.7
        # Low values during off-season
        return 0.2
    
    # Southern hemisphere basins peak in January-February
    else:  # SI, SP, etc.
        # Peak: December-April, maximum in January-February
        if current_month <= 4 or current_month >= 12:
            # Higher values during peak months (Jan-Feb)
            if current_month == 1 or current_month == 2:
                return 0.9
            # Moderate values during normal season months
            return 0.7
        # Low values during off-season
        return 0.2

def get_basin_formation_factor(basin):
    """Get basin-specific factor for storm formation frequency."""
    # Historical formation frequency by basin
    basin_factors = {
        'WP': 0.9,  # Western Pacific has most formations
        'EP': 0.7,  # Eastern Pacific
        'NA': 0.7,  # North Atlantic
        'NI': 0.5,  # North Indian
        'SI': 0.6,  # South Indian
        'SP': 0.6,  # South Pacific
        'SA': 0.2,  # South Atlantic (very rare)
    }
    return basin_factors.get(basin, 0.5)  # Default for unknown basin

def predict_formation_intensity(state, prediction):
    """Predict the initial intensity if a storm forms."""
    sst = state.get("sea_surface_temp", {}).get("value", 28.0)
    predicted_pressure = prediction.get("pressure", 1010)
    
    # Very warm waters can support stronger initial formations
    if sst >= 29.5 and predicted_pressure < 1005:
        return "TS"  # Tropical Storm
    elif sst >= 28.5 and predicted_pressure < 1008:
        # 50/50 chance of forming as TS vs TD when conditions are favorable
        return "TS" if random.random() > 0.5 else "TD"
    else:
        return "TD"  # Tropical Depression
