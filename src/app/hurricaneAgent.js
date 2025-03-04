// hurricaneAgent.js

/**
 * Hurricane Prediction Agent Environment
 * This class provides an environment for a reinforcement learning agent
 * to predict hurricane trajectories and intensities.
 */
class HurricaneEnvironment {
    constructor() {
      this.hurricaneData = []; // Historical hurricane data
      this.nasaData = {};      // NASA data layers (sea surface temp, etc.)
      this.currentState = null;
      this.timeStep = 0;       // Current time step in the simulation
      this.history = [];       // Track agent predictions for evaluation
    }
  
    /**
     * Initialize the environment with data
     */
    async initialize(historicalData, nasaDataService) {
      this.hurricaneData = historicalData;
      this.nasaData = nasaDataService;
      this.reset();
      return this.getState();
    }
  
    /**
     * Reset the environment to initial state for a new episode
     */
    reset() {
      // Start with a random historical hurricane at its beginning
      const randomIndex = Math.floor(Math.random() * this.hurricaneData.length);
      const selectedHurricane = this.hurricaneData[randomIndex];
      
      this.currentState = {
        hurricaneId: selectedHurricane.id,
        position: selectedHurricane.initialPosition,
        windSpeed: selectedHurricane.initialWindSpeed,
        pressure: selectedHurricane.initialPressure,
        seaSurfaceTemp: this.getNasaDataForLocation(selectedHurricane.initialPosition),
        timestamp: selectedHurricane.startTime,
        actual: selectedHurricane.track // Full track data for evaluation
      };
      
      this.timeStep = 0;
      this.history = [];
      
      return this.getState();
    }
  
    /**
     * Get NASA data for a specific location
     */
    getNasaDataForLocation(position) {
      // Implementation would call the NASA service to get relevant data
      // such as sea surface temperature for the given coordinates
      return {};
    }
  
    /**
     * Get the current state observation for the agent
     */
    getState() {
      return {
        position: this.currentState.position,
        windSpeed: this.currentState.windSpeed,
        pressure: this.currentState.pressure,
        seaSurfaceTemp: this.currentState.seaSurfaceTemp,
        timestamp: this.currentState.timestamp,
        timeStep: this.timeStep
      };
    }
  
    /**
     * Take an action in the environment
     * @param {Object} action - Predicted position and intensity
     * @returns {Object} - New state, reward, done flag
     */
    step(action) {
      // Record the agent's prediction
      this.history.push({
        timeStep: this.timeStep,
        prediction: action,
        actual: this.getActualForTimeStep(this.timeStep + 1)
      });
      
      // Update the time step
      this.timeStep++;
      
      // Update the current state to actual values for next time step
      const actualNextState = this.getActualForTimeStep(this.timeStep);
      if (actualNextState) {
        this.currentState = {
          ...this.currentState,
          position: actualNextState.position,
          windSpeed: actualNextState.windSpeed,
          pressure: actualNextState.pressure,
          seaSurfaceTemp: this.getNasaDataForLocation(actualNextState.position),
          timestamp: actualNextState.timestamp
        };
      }
      
      // Calculate reward based on prediction accuracy
      const reward = this.calculateReward(action, actualNextState);
      
      // Check if episode is done
      const done = !actualNextState || this.timeStep >= this.currentState.actual.length - 1;
      
      return {
        state: this.getState(),
        reward,
        done
      };
    }
  
    /**
     * Get actual hurricane state for a specific time step
     */
    getActualForTimeStep(timeStep) {
      if (timeStep >= this.currentState.actual.length) {
        return null;
      }
      return this.currentState.actual[timeStep];
    }
  
    /**
     * Calculate reward based on prediction accuracy
     */
    calculateReward(prediction, actual) {
      if (!actual) return 0;
      
      // Calculate positional error (in km)
      const positionError = this.calculateDistance(
        prediction.position,
        actual.position
      );
      
      // Calculate intensity error
      const intensityError = Math.abs(prediction.windSpeed - actual.windSpeed);
      
      // Reward decreases with error
      const positionReward = Math.max(0, 100 - positionError);
      const intensityReward = Math.max(0, 50 - intensityError);
      
      return positionReward + intensityReward;
    }
  
    /**
     * Calculate distance between two geographic points (in km)
     */
    calculateDistance(pos1, pos2) {
      // Haversine formula for calculating distance between two points on Earth
      const R = 6371; // Earth's radius in km
      const dLat = this.deg2rad(pos2.lat - pos1.lat);
      const dLon = this.deg2rad(pos2.lon - pos1.lon);
      const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(this.deg2rad(pos1.lat)) * Math.cos(this.deg2rad(pos2.lat)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
      const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      return R * c;
    }
  
    deg2rad(deg) {
      return deg * (Math.PI / 180);
    }
  
    /**
     * Evaluate agent performance over an episode
     */
    evaluatePerformance() {
      // Calculate average position and intensity errors
      const errors = this.history.map(record => {
        if (!record.actual) return null;
        
        const posError = this.calculateDistance(
          record.prediction.position,
          record.actual.position
        );
        
        const intensityError = Math.abs(
          record.prediction.windSpeed - record.actual.windSpeed
        );
        
        return { posError, intensityError };
      }).filter(error => error !== null);
      
      if (errors.length === 0) return { avgPosError: Infinity, avgIntensityError: Infinity };
      
      const avgPosError = errors.reduce((sum, err) => sum + err.posError, 0) / errors.length;
      const avgIntensityError = errors.reduce((sum, err) => sum + err.intensityError, 0) / errors.length;
      
      return { avgPosError, avgIntensityError };
    }
  }
  
  /**
   * Prediction Agent using a simple model
   */
  class HurricanePredictionAgent {
    constructor() {
      this.weights = {
        position: {
          lat: Array(5).fill(0.1),  // Weights for previous 5 positions
          lon: Array(5).fill(0.1)
        },
        intensity: Array(5).fill(0.1)
      };
      this.learningRate = 0.01;
    }
  
    /**
     * Make a prediction based on current state and history
     */
    predict(state, history) {
      // Extract the last 5 positions (or fewer if not available)
      const positions = history
        .slice(-5)
        .map(h => h.state.position)
        .reverse();
      
      positions.push(state.position);
      
      // Extract the last 5 wind speeds
      const windSpeeds = history
        .slice(-5)
        .map(h => h.state.windSpeed)
        .reverse();
      
      windSpeeds.push(state.windSpeed);
      
      // Predict next position using weighted average of position changes
      let latPredict = state.position.lat;
      let lonPredict = state.position.lon;
      
      // Calculate position changes and apply weights
      for (let i = 1; i < positions.length; i++) {
        const idx = i - 1;
        if (idx < this.weights.position.lat.length) {
          latPredict += (positions[i].lat - positions[i-1].lat) * this.weights.position.lat[idx];
          lonPredict += (positions[i].lon - positions[i-1].lon) * this.weights.position.lon[idx];
        }
      }
      
      // Predict intensity (wind speed)
      let windPredict = state.windSpeed;
      for (let i = 1; i < windSpeeds.length; i++) {
        const idx = i - 1;
        if (idx < this.weights.intensity.length) {
          windPredict += (windSpeeds[i] - windSpeeds[i-1]) * this.weights.intensity[idx];
        }
      }
      
      // Adjust prediction based on environmental factors
      if (state.seaSurfaceTemp && state.seaSurfaceTemp.value > 28) {
        // Warmer water intensifies hurricanes
        windPredict *= 1.05;
      }
      
      // Additional factors could be considered here (wind shear, etc.)
      
      return {
        position: { lat: latPredict, lon: lonPredict },
        windSpeed: windPredict
      };
    }
  
    /**
     * Update the model based on reward and actual outcomes
     */
    update(prediction, actual, reward) {
      if (!actual) return;
      
      // Update position weights
      const latError = actual.position.lat - prediction.position.lat;
      const lonError = actual.position.lon - prediction.position.lon;
      
      for (let i = 0; i < this.weights.position.lat.length; i++) {
        this.weights.position.lat[i] += this.learningRate * latError * reward;
        this.weights.position.lon[i] += this.learningRate * lonError * reward;
      }
      
      // Update intensity weights
      const intensityError = actual.windSpeed - prediction.windSpeed;
      for (let i = 0; i < this.weights.intensity.length; i++) {
        this.weights.intensity[i] += this.learningRate * intensityError * reward;
      }
    }
  
    /**
     * Train the agent on historical data
     */
    async train(environment, episodes = 100) {
      const performance = [];
      
      for (let episode = 0; episode < episodes; episode++) {
        const state = environment.reset();
        let done = false;
        const history = [];
        
        while (!done) {
          // Get current state
          const currentState = environment.getState();
          
          // Make prediction
          const action = this.predict(currentState, history);
          
          // Take action and observe result
          const { state: nextState, reward, done: isDone } = environment.step(action);
          
          // Store state for history
          history.push({ state: currentState, action });
          
          // Update model
          const actual = environment.getActualForTimeStep(environment.timeStep);
          this.update(action, actual, reward);
          
          done = isDone;
        }
        
        // Evaluate performance
        const episodePerformance = environment.evaluatePerformance();
        performance.push(episodePerformance);
        
        console.log(`Episode ${episode + 1}/${episodes} - Avg Position Error: ${episodePerformance.avgPosError.toFixed(2)} km, Avg Intensity Error: ${episodePerformance.avgIntensityError.toFixed(2)} mph`);
      }
      
      return performance;
    }
  }
  
  export { HurricaneEnvironment, HurricanePredictionAgent };