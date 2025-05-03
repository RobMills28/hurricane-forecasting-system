/**
 * Hurricane Agent API - Client-side connector
 * 
 * This module provides functions to interact with the Python hurricane prediction backend.
 */

// Base URL for the API - adjust based on your deployment
const API_BASE_URL = process.env.NEXT_PUBLIC_HURRICANE_API_URL || 'http://localhost:8000';

/**
 * Start training a hurricane prediction agent
 * 
 * @param {Object} options - Training options
 * @param {string} options.agent_id - Unique ID for the agent
 * @param {number} [options.episodes=100] - Number of training episodes
 * @param {string} [options.basin] - Optional basin to focus on
 * @param {boolean} [options.use_basin_models=true] - Whether to use basin-specific models
 * @param {boolean} [options.use_dynamic_weights=true] - Whether to use dynamic weights
 * @param {number} [options.ensemble_size=5] - Number of ensemble members
 * @param {number} [options.min_category=0] - Minimum hurricane category to include
 * @returns {Promise<Object>} Training start status
 */
export async function startTraining(options) {
  try {
    const response = await fetch(`${API_BASE_URL}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(options)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to start training: ${response.status} - ${errorText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error starting training:', error);
    throw error;
  }
}

/**
 * Check the status of a training job
 * 
 * @param {string} agentId - The agent ID to check
 * @returns {Promise<Object>} Training status
 */
export async function getTrainingStatus(agentId) {
  try {
    const response = await fetch(`${API_BASE_URL}/training_status/${agentId}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to get training status: ${response.status} - ${errorText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting training status:', error);
    throw error;
  }
}

/**
 * Make a prediction using a trained agent
 * 
 * @param {string} agentId - ID of the trained agent
 * @param {Object} currentState - Current hurricane state
 * @param {Array} [history=[]] - History of previous states
 * @returns {Promise<Object>} Prediction result
 */
export async function makePrediction(agentId, currentState, history = []) {
  try {
    console.log("Making RL prediction request to API with agent:", agentId);
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        agent_id: agentId,
        current_state: currentState,
        history
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to make prediction: ${response.status} - ${errorText}`);
    }
    
    const result = await response.json();
    console.log("RL prediction result from API:", result);
    
    return result;
  } catch (error) {
    console.error('Error making RL prediction:', error);
    throw error;
  }
}

/**
 * List all trained agents
 * 
 * @returns {Promise<Object>} List of trained agents
 */
export async function listTrainedAgents() {
  try {
    const response = await fetch(`${API_BASE_URL}/trained_agents`);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to list agents: ${response.status} - ${errorText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error listing trained agents:', error);
    throw error;
  }
}

/**
 * Delete a trained agent
 * 
 * @param {string} agentId - The agent ID to delete
 * @returns {Promise<Object>} Deletion result
 */
export async function deleteAgent(agentId) {
  try {
    const response = await fetch(`${API_BASE_URL}/agent/${agentId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to delete agent: ${response.status} - ${errorText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error deleting agent:', error);
    throw error;
  }
}

/**
 * Check if the API is running
 * 
 * @returns {Promise<boolean>} True if API is accessible
 */
export async function checkApiHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    return response.ok;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
}