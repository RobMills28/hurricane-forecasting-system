"""
API server for hurricane prediction models

This module provides an interface between the JavaScript frontend and Python backend.
"""

import asyncio
import json
import os
import logging
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our hurricane agents components
from .environment import HurricaneEnvironment
from .agent import HurricanePredictionAgent
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

# Background task to train an agent
async def train_agent_task(agent_id: str, options: Dict[str, Any]):
    """Background task to train an agent."""
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
        }
        agent = HurricanePredictionAgent(agent_options)
        
        # Train agent
        training_tasks[agent_id]["status"] = "training"
        episodes = options.get("episodes", 100)
        
        # Update progress as training proceeds
        for episode in range(1, episodes + 1):
            # Reset for this episode
            environment.reset()
            
            done = False
            history = []
            
            # Episode loop
            while not done:
                current_state = environment.get_state()
                action = agent.predict(current_state, history)
                result = environment.step(action)
                next_state = result.get("state", {})
                reward = result.get("reward", 0)
                done = result.get("done", False)
                
                history.append({"state": current_state, "action": action})
                
                actual = environment.get_actual_for_time_step(environment.time_step)
                agent.update(action, actual, reward)
            
            # Evaluate performance
            episode_performance = environment.evaluate_performance()
            
            # Update progress
            training_tasks[agent_id]["progress"] = episode / episodes
            training_tasks[agent_id]["metrics"] = episode_performance
            
            # Log progress every 10 episodes
            if episode % 10 == 0 or episode == 1 or episode == episodes:
                logger.info(
                    f"Episode {episode}/{episodes} - "
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
    prediction = agent.predict(request.current_state, request.history)
    
    # Add hurricane category based on wind speed
    if "wind_speed" in prediction:
        prediction["category"] = get_hurricane_category(prediction["wind_speed"])
    
    return prediction

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