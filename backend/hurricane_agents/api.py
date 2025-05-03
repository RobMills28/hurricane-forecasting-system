"""
API server for hurricane prediction models

This module provides an interface between the JavaScript frontend and Python backend.
"""
import torch
from .agent import HurricanePredictionAgent, ReplayBuffer
import pandas as pd
from .utils import determine_basin
import asyncio
import json
import os
import logging
import random
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
            "state_dim": 10,  # Adjust based on state representation
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
        logger.info(f"Coordinates type: {type(request.coordinates)}")
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
        
        # Get weather observations
        observations = await get_storm_observations(lat, lon, request.data_source, request.basin)
        logger.info(f"Got observations: {observations}")
        
        # Create state object for prediction
        current_state = {
            "position": {"lat": lat, "lon": lon},
            "wind_speed": observations.get("windSpeed", 75),
            "pressure": observations.get("barometricPressure", 990),
            "basin": request.basin,
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
    This uses the agent system to predict areas where new storms might form.
    """
    try:
        logger.info("Fetching potential storm formation areas")
        
        # Get global environmental data (sea surface temperatures, pressure systems, etc.)
        environmental_data = await get_global_environmental_data()
        
        # Use the agent system to predict potential formation areas
        potential_areas = predict_potential_storm_areas(environmental_data)
        
        return {"potential_areas": potential_areas}
    except Exception as e:
        logger.error(f"Error in get_potential_storm_areas: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return simulated data for testing if the actual prediction fails
        return {
            "potential_areas": generate_simulated_formation_areas()
        }

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
    # In a full implementation, this would:
    # 1. Use the trained HurricanePredictionAgent to analyze environmental data
    # 2. Apply basin-specific models to identify favorable formation conditions
    # 3. Calculate formation probabilities based on historical patterns
    
    potential_areas = []
    
    # Check if we have trained agents available
    if "default-hurricane-agent" in trained_agents:
        agent = trained_agents["default-hurricane-agent"]
        
        # Process environmental data to predict potential formations
        pressure_systems = environmental_data.get("pressure_systems", [])
        sst_data = environmental_data.get("sea_surface_temps", {})
        wind_shear = environmental_data.get("wind_shear", {})
        
        for system in pressure_systems:
            # Only consider low pressure systems
            if system.get("type") != "low":
                continue
                
            position = system.get("position")
            if not position:
                continue
                
            # Determine basin based on position
            lat, lon = position
            basin = determine_basin(lon, lat)
            
            # Calculate formation probability based on conditions
            basin_sst = get_basin_sst(basin, sst_data)
            basin_wind_shear = get_basin_wind_shear(basin, wind_shear)
            pressure = system.get("pressure", 1013)
            
            probability = calculate_formation_probability(
                pressure, basin_sst, basin_wind_shear, lat, lon
            )
            
            # Predict potential intensity if formation occurs
            intensity = predict_formation_intensity(basin_sst, pressure)
            
            # Add to potential areas if probability exceeds threshold
            if probability > 0.2:  # 20% minimum threshold
                potential_areas.append({
                    "id": f"pot_{basin}_{len(potential_areas)}",
                    "position": position,
                    "probability": probability,
                    "basin": basin,
                    "intensity": intensity
                })
    
    # If no areas found from the model, return simulated data for testing
    if not potential_areas:
        potential_areas = generate_simulated_formation_areas()
        
    return potential_areas

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

def calculate_formation_probability(pressure, sst, wind_shear, lat, lon):
    """
    Calculate the probability of storm formation based on environmental factors.
    """
    # Pressure factor - lower pressure is more favorable
    pressure_factor = max(0, min(1, (1013 - pressure) / 15))
    
    # SST factor - higher SST is more favorable (above 26°C)
    sst_factor = max(0, min(1, (sst - 26) / 4)) if sst > 26 else 0
    
    # Latitude factor - formations typically occur between 5-20° latitude
    abs_lat = abs(lat)
    lat_factor = 0
    if 5 <= abs_lat <= 20:
        lat_factor = 1.0
    elif abs_lat < 5:
        lat_factor = abs_lat / 5
    elif 20 < abs_lat <= 30:
        lat_factor = max(0, (30 - abs_lat) / 10)
    
    # Season factor (based on hemisphere and current month)
    month = pd.Timestamp.now().month
    season_factor = 0.5  # Default
    
    if lat > 0:  # Northern hemisphere
        # Peak season: June-November
        if 6 <= month <= 11:
            season_factor = 0.9
    else:  # Southern hemisphere
        # Peak season: November-April
        if month >= 11 or month <= 4:
            season_factor = 0.9
    
    # Combine factors with weights
    combined = (
        pressure_factor * 0.3 +
        sst_factor * 0.3 +
        wind_shear * 0.2 +
        lat_factor * 0.1 +
        season_factor * 0.1
    )
    
    # Add some randomness to simulate prediction uncertainty
    random_factor = 1.0 + (random.random() * 0.2 - 0.1)  # ±10%
    
    return min(0.95, max(0.1, combined * random_factor))

def predict_formation_intensity(sst, pressure):
    """
    Predict the initial intensity if a storm forms.
    """
    # Very warm waters can support stronger initial formations
    if sst >= 29 and pressure < 1005:
        return "TS"  # Tropical Storm
    else:
        return "TD"  # Tropical Depression

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
    current_month = pd.Timestamp.now().month
    active_regions = []
    for region in formation_regions:
        basin = region["basin"]
        # Northern hemisphere hurricane season: June-November
        # Southern hemisphere hurricane season: November-April
        if (basin in ["NA", "EP", "WP", "NI"] and 6 <= current_month <= 11) or \
           (basin in ["SI", "SP"] and (current_month >= 11 or current_month <= 4)):
            active_regions.append(region)
    
    # If no active regions based on season, return some anyway for testing
    if not active_regions:
        active_regions = formation_regions[:5]
    
    # Generate potential formation areas with simulated probabilities
    potential_areas = []
    for i, region in enumerate(active_regions):
        # Random probability, but higher for main formation regions
        probability = 0.3 + (random.random() * 0.5)
        
        # Random intensity based on probability
        intensity = "TS" if probability > 0.6 else "TD"
        
        # Add variation to the position for more realism
        lat_jitter = (random.random() * 4) - 2  # ±2 degrees
        lon_jitter = (random.random() * 4) - 2  # ±2 degrees
        
        lat = region["position"][0] + lat_jitter
        lon = region["position"][1] + lon_jitter
        
        potential_areas.append({
            "id": f"pot_{i}",
            "position": [lat, lon],
            "probability": probability,
            "basin": region["basin"],
            "intensity": intensity
        })
    
    # Return a subset to avoid too many points
    return random.sample(potential_areas, min(5, len(potential_areas)))