// hurricaneAgent.js

/**
 * Enhanced Hurricane Prediction Agent Environment
 * This class provides an environment for advanced hurricane trajectory and intensity prediction
 * using real-world data and sophisticated modeling techniques.
 */
class HurricaneEnvironment {
  constructor() {
    this.hurricaneData = []; // Historical hurricane data
    this.nasaData = {};      // NASA data layers (sea surface temp, etc.)
    this.currentState = null;
    this.timeStep = 0;       // Current time step in the simulation
    this.history = [];       // Track agent predictions for evaluation
    this.basinModels = {};   // Basin-specific models for regional specialization
  }

  /**
   * Initialize the environment with data
   */
  async initialize(historicalData, nasaDataService) {
    this.hurricaneData = historicalData;
    this.nasaData = nasaDataService;
    
    // Organize data by basin for basin-specific training
    this.organizeDataByBasin();
    
    this.reset();
    return this.getState();
  }

  /**
   * Organize hurricane data by basin for specialized training
   */
  organizeDataByBasin() {
    const basinData = {};
    
    this.hurricaneData.forEach(hurricane => {
      const basin = hurricane.basin || 'UNKNOWN';
      if (!basinData[basin]) {
        basinData[basin] = [];
      }
      basinData[basin].push(hurricane);
    });
    
    this.basinData = basinData;
    
    // Log statistics about data distribution
    const basinStats = Object.entries(basinData)
      .map(([basin, data]) => `${basin}: ${data.length} storms`)
      .join(', ');
    console.log(`Hurricane data by basin: ${basinStats}`);
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
      name: selectedHurricane.name,
      basin: selectedHurricane.basin,
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
    if (!position || !this.nasaData.getGibsLayers) {
      // Default values if NASA data service not available
      return {
        type: 'seaSurfaceTemperature',
        value: this.estimateSeaSurfaceTempForLocation(position)
      };
    }
    
    // In a complete implementation, this would query NASA data for the coordinates
    try {
      // Estimate SST based on position and climate patterns
      return {
        type: 'seaSurfaceTemperature',
        value: this.estimateSeaSurfaceTempForLocation(position)
      };
    } catch (error) {
      console.error('Error getting NASA data:', error);
      return {
        type: 'seaSurfaceTemperature',
        value: 28 // Default fallback
      };
    }
  }
  
  /**
   * Estimate sea surface temperature for a location based on position and season
   * This is a simplified model until we integrate real-time NASA data
   */
  estimateSeaSurfaceTempForLocation(position) {
    if (!position) return 28; // Default value
    
    const { lat, lon } = position;
    
    // Base temperature based on absolute latitude (warmer near equator)
    const latitudeEffect = 30 - (Math.abs(lat) * 0.3);
    
    // Longitude effects - customized by ocean basin
    let basinEffect = 0;
    
    // Simplified basin determination
    let basin = 'UNKNOWN';
    if (lon > -100 && lon < 0 && lat > 0) basin = 'NA'; // North Atlantic
    else if (lon >= -180 && lon < -100 && lat > 0) basin = 'EP'; // Eastern Pacific
    else if (lon >= 100 && lon < 180 && lat > 0) basin = 'WP'; // Western Pacific
    else if (lon >= 40 && lon < 100 && lat > 0) basin = 'NI'; // North Indian
    else if (lon >= 40 && lon < 135 && lat <= 0) basin = 'SI'; // South Indian
    else if (lon >= 135 && lon < 180 && lat <= 0) basin = 'SP'; // South Pacific
    
    // Basin-specific effects
    switch (basin) {
      case 'WP': // Western Pacific tends to be warmer
        basinEffect = 1.5;
        break;
      case 'NI': // North Indian Ocean
        basinEffect = 2.0;
        break;
      case 'NA': // North Atlantic
        basinEffect = 0.0;
        break;
      default:
        basinEffect = 0.5;
    }
    
    // Get current month for seasonal effects
    const currentMonth = this.currentState?.timestamp?.getMonth() || (new Date()).getMonth();
    
    // Seasonal effect (northern and southern hemisphere have opposite seasons)
    let seasonalEffect = 0;
    if (lat > 0) { // Northern hemisphere
      // Warmer in northern summer (Jun-Aug)
      seasonalEffect = Math.sin((currentMonth - 5) * Math.PI / 6) * 2;
    } else { // Southern hemisphere
      // Warmer in southern summer (Dec-Feb)
      seasonalEffect = Math.sin((currentMonth - 11) * Math.PI / 6) * 2;
    }
    
    // Calculate final temperature with some randomness
    const temperature = latitudeEffect + basinEffect + seasonalEffect + (Math.random() - 0.5);
    
    // Constrain to realistic values (20-32°C)
    return Math.max(20, Math.min(32, temperature));
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
      basin: this.currentState.basin,
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
    if (!this.currentState.actual || timeStep >= this.currentState.actual.length) {
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
    
    // Calculate pressure error
    const pressureError = Math.abs(prediction.pressure - actual.pressure);
    
    // Reward decreases with error
    const positionReward = Math.max(0, 100 - positionError);
    const intensityReward = Math.max(0, 50 - intensityError);
    const pressureReward = Math.max(0, 30 - (pressureError / 2));
    
    // Weight rewards based on forecast time
    // Position accuracy is more important for short-term forecasts
    // Intensity becomes more important for medium-term forecasts
    const forecastHour = this.timeStep * 6; // Assuming 6-hour time steps
    let positionWeight, intensityWeight, pressureWeight;
    
    if (forecastHour <= 24) {
      // Short-term (0-24h): Position is most critical
      positionWeight = 0.6;
      intensityWeight = 0.3;
      pressureWeight = 0.1;
    } else if (forecastHour <= 72) {
      // Medium-term (24-72h): Balance position and intensity
      positionWeight = 0.5;
      intensityWeight = 0.4;
      pressureWeight = 0.1;
    } else {
      // Long-term (>72h): Intensity trend becomes more important
      positionWeight = 0.4;
      intensityWeight = 0.5;
      pressureWeight = 0.1;
    }
    
    return (
      positionReward * positionWeight + 
      intensityReward * intensityWeight + 
      pressureReward * pressureWeight
    );
  }

  /**
   * Calculate distance between two geographic points (in km)
   */
  calculateDistance(pos1, pos2) {
    if (!pos1 || !pos2) return 1000; // Large error if positions missing
    
    // Haversine formula for calculating distance between two points on Earth
    const R = 6371; // Earth's radius in km
    const dLat = this.deg2rad(pos2.lat - pos1.lat);
    const dLon = this.deg2rad(pos2.lon - pos1.lon);
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(this.deg2rad(pos1.lat)) * Math.cos(this.deg2rad(pos2.lat)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
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
    
    // Safely calculate position error
      const posError = this.calculateDistance(
        record.prediction.position,
        record.actual.position
      );
    
    // Safely calculate intensity error
      const intensityError = Math.abs(
        (record.prediction.windSpeed || 0) - (record.actual.windSpeed || 0)
      );
    
    // Safely calculate pressure error
      const pressureError = Math.abs(
        (record.prediction.pressure || 1000) - (record.actual.pressure || 1000)
      );
    
    // Calculate error by forecast period
      const timeStep = record.timeStep;
      const forecastHour = timeStep * 6; // Assuming 6-hour time steps
    
      let forecastPeriod;
      if (forecastHour <= 24) forecastPeriod = '24h';
      else if (forecastHour <= 48) forecastPeriod = '48h';
      else if (forecastHour <= 72) forecastPeriod = '72h';
      else if (forecastHour <= 96) forecastPeriod = '96h';
      else forecastPeriod = '120h';
    
      return { 
        posError: isNaN(posError) ? 0 : posError, 
        intensityError: isNaN(intensityError) ? 0 : intensityError, 
        pressureError: isNaN(pressureError) ? 0 : pressureError,
        forecastHour,
        forecastPeriod
      };
    }).filter(error => error !== null);
  
  // Default values if no valid errors
    if (errors.length === 0) {
      return { 
        avgPosError: 0, 
        avgIntensityError: 0,
        avgPressureError: 0,
        byForecastPeriod: {}
      };
    }
  
  // Safely calculate averages
    const sum = (arr) => arr.reduce((a, b) => a + b, 0);
    const safeAvg = (arr) => arr.length > 0 ? sum(arr) / arr.length : 0;
  
  // Calculate overall averages - prevent NaN
    const avgPosError = safeAvg(errors.map(e => e.posError));
    const avgIntensityError = safeAvg(errors.map(e => e.intensityError));
    const avgPressureError = safeAvg(errors.map(e => e.pressureError));
  
  // Group by forecast period with safety checks
    const byForecastPeriod = {};
    const periods = ['24h', '48h', '72h', '96h', '120h'];
  
    periods.forEach(period => {
      const periodErrors = errors.filter(err => err.forecastPeriod === period);
      if (periodErrors.length === 0) {
        byForecastPeriod[period] = null;
        return;
      }
    
      byForecastPeriod[period] = {
        count: periodErrors.length,
        avgPosError: safeAvg(periodErrors.map(e => e.posError)),
        avgIntensityError: safeAvg(periodErrors.map(e => e.intensityError)),
        avgPressureError: safeAvg(periodErrors.map(e => e.pressureError))
      };
    });
  
    return { 
      avgPosError: isNaN(avgPosError) ? 0 : avgPosError, 
      avgIntensityError: isNaN(avgIntensityError) ? 0 : avgIntensityError,
      avgPressureError: isNaN(avgPressureError) ? 0 : avgPressureError,
      byForecastPeriod
    };
  }
}
/**
 * Advanced Hurricane Prediction Agent using ensemble methods
 * and basin-specific models
 */
class HurricanePredictionAgent {
  constructor(options = {}) {
    // Configuration options
    this.options = {
      useDynamicWeights: true,    // Use dynamic weights based on storm phase
      useBasinModels: true,       // Use basin-specific models
      ensembleSize: 5,            // Number of sub-models in ensemble
      ...options
    };
    
    // Initialize weights for trajectory prediction
    this.weights = {
      position: {
        lat: Array(5).fill(0.1),  // Weights for previous 5 positions
        lon: Array(5).fill(0.1)
      },
      intensity: Array(5).fill(0.1),
      pressure: Array(5).fill(0.1)
    };
    
    // Basin-specific models
    this.basinModels = {
      NA: { ...this.createInitialWeights() },  // North Atlantic
      EP: { ...this.createInitialWeights() },  // Eastern Pacific
      WP: { ...this.createInitialWeights() },  // Western Pacific
      NI: { ...this.createInitialWeights() },  // North Indian
      SI: { ...this.createInitialWeights() },  // South Indian
      SP: { ...this.createInitialWeights() },  // South Pacific
      DEFAULT: { ...this.createInitialWeights() }  // Default model
    };
    
    // Phase-specific weight adjustments
    this.phaseAdjustments = {
      early: {
        position: { factor: 1.2 },     // Early phase: directional persistence is stronger
        intensity: { factor: 1.5 }     // Early phase: rapid intensification possible
      },
      peak: {
        position: { factor: 1.0 },     // Peak phase: more balanced
        intensity: { factor: 0.8 }     // Peak phase: intensity changes less dramatically
      },
      late: {
        position: { factor: 0.7 },     // Late phase: track becomes more variable
        intensity: { factor: 1.3 }     // Late phase: rapid weakening possible
      }
    };
    
    // Environmental factor weights
    this.environmentalFactors = {
      seaSurfaceTemp: 0.3,              // Influence of SST on intensification
      latitudinalEffect: 0.2,           // Higher latitudes encourage weakening/recurvature
      seasonalEffect: 0.1               // Seasonal patterns
    };
    
    this.learningRate = 0.01;
    this.ensembleMembers = this.initializeEnsemble();
    
    // Performance tracking
    this.trainingPerformance = [];
  }
  
  /**
   * Create initial weights object
   */
  createInitialWeights() {
    return {
      position: {
        lat: Array(5).fill(0.1).map(() => 0.1 + (Math.random() * 0.05 - 0.025)),
        lon: Array(5).fill(0.1).map(() => 0.1 + (Math.random() * 0.05 - 0.025))
      },
      intensity: Array(5).fill(0.1).map(() => 0.1 + (Math.random() * 0.05 - 0.025)),
      pressure: Array(5).fill(0.1).map(() => 0.1 + (Math.random() * 0.05 - 0.025))
    };
  }
  
  /**
   * Initialize ensemble models
   * Creates slightly different variations of the model for ensemble forecasting
   */
  initializeEnsemble() {
    const ensemble = [];
    
    for (let i = 0; i < this.options.ensembleSize; i++) {
      // Create variation of weights
      const variationFactor = 0.1 + (Math.random() * 0.1);
      const weights = {};
      
      // Deep copy with variations
      Object.keys(this.weights).forEach(key => {
        if (typeof this.weights[key] === 'object' && !Array.isArray(this.weights[key])) {
          weights[key] = {};
          Object.keys(this.weights[key]).forEach(subKey => {
            weights[key][subKey] = this.weights[key][subKey].map(w => 
              w * (1 + (Math.random() * variationFactor * 2 - variationFactor))
            );
          });
        } else if (Array.isArray(this.weights[key])) {
          weights[key] = this.weights[key].map(w => 
            w * (1 + (Math.random() * variationFactor * 2 - variationFactor))
          );
        } else {
          weights[key] = this.weights[key];
        }
      });
      
      ensemble.push({
        weights,
        environmentalFactors: { ...this.environmentalFactors },
        learningRate: this.learningRate * (1 + (Math.random() * 0.2 - 0.1))
      });
    }
    
    return ensemble;
  }

  /**
   * Make a prediction based on current state and history
   */
  predict(state, history) {
    // Get basin-specific model if available
    const basin = state.basin || 'DEFAULT';
    const weights = this.options.useBasinModels && this.basinModels[basin] 
      ? this.basinModels[basin] 
      : this.weights;
      
    // Make individual ensemble predictions
    const ensemblePredictions = this.ensembleMembers.map(member => 
      this.makeSinglePrediction(state, history, member.weights, member.environmentalFactors)
    );
    
    // Add main model prediction
    ensemblePredictions.push(
      this.makeSinglePrediction(state, history, weights, this.environmentalFactors)
    );
    
    // Combine ensemble predictions
    return this.combineEnsemblePredictions(ensemblePredictions, state);
  }
  
  /**
   * Make a single model prediction
   */
  makeSinglePrediction(state, history, weights, environmentalFactors) {
    // Get storm phase based on history length
    const stormPhase = this.determineStormPhase(history);
    
    // Extract the last positions (or fewer if not available)
    const positions = history
      .slice(-5)
      .map(h => h.state.position)
      .reverse();
    
    positions.push(state.position);
    
    // Extract the last wind speeds
    const windSpeeds = history
      .slice(-5)
      .map(h => h.state.windSpeed)
      .reverse();
    
    windSpeeds.push(state.windSpeed);
    
    // Extract the last pressures
    const pressures = history
      .slice(-5)
      .map(h => h.state.pressure)
      .reverse();
    
    pressures.push(state.pressure);
    
    // Adjust weights based on storm phase if enabled
    let adjustedWeights = { ...weights };
    if (this.options.useDynamicWeights && stormPhase) {
      adjustedWeights = this.adjustWeightsForPhase(weights, stormPhase);
    }
    
    // Predict next position using weighted average of position changes
    let latPredict = state.position.lat;
    let lonPredict = state.position.lon;
    
    // Calculate position changes and apply weights
    for (let i = 1; i < positions.length; i++) {
      const idx = i - 1;
      if (idx < adjustedWeights.position.lat.length) {
        latPredict += (positions[i].lat - positions[i-1].lat) * adjustedWeights.position.lat[idx];
        lonPredict += (positions[i].lon - positions[i-1].lon) * adjustedWeights.position.lon[idx];
      }
    }
    
    // Predict intensity (wind speed)
    let windPredict = state.windSpeed;
    for (let i = 1; i < windSpeeds.length; i++) {
      const idx = i - 1;
      if (idx < adjustedWeights.intensity.length) {
        windPredict += (windSpeeds[i] - windSpeeds[i-1]) * adjustedWeights.intensity[idx];
      }
    }
    
    // Predict pressure
    let pressurePredict = state.pressure;
    for (let i = 1; i < pressures.length; i++) {
      const idx = i - 1;
      if (idx < adjustedWeights.pressure.length) {
        pressurePredict += (pressures[i] - pressures[i-1]) * adjustedWeights.pressure[idx];
      }
    }
    
    // Apply environmental factors
    // Sea surface temperature effect
    if (state.seaSurfaceTemp && state.seaSurfaceTemp.value) {
      // SST effect on intensity
      const sstEffect = this.calculateSSTEffect(state.seaSurfaceTemp.value);
      windPredict *= (1 + (sstEffect * environmentalFactors.seaSurfaceTemp));
      
      // Higher SST typically means lower pressure
      pressurePredict *= (1 - (sstEffect * environmentalFactors.seaSurfaceTemp * 0.05));
    }
    
    // Latitude effect (higher latitudes typically mean weakening and recurvature)
    const latitude = state.position.lat;
    const latitudeEffect = this.calculateLatitudeEffect(latitude, state.basin);
    
    // Apply latitude effect to track and intensity
    if (Math.abs(latitude) > 25) {
      // More poleward movement at higher latitudes
      const polewardDirection = latitude > 0 ? 1 : -1;
      latPredict += latitudeEffect * polewardDirection * environmentalFactors.latitudinalEffect;
      
      // Generally eastward movement at higher latitudes
      lonPredict += latitudeEffect * 0.2 * environmentalFactors.latitudinalEffect;
      
      // Weakening at higher latitudes
      windPredict *= (1 - (latitudeEffect * environmentalFactors.latitudinalEffect * 0.1));
      pressurePredict *= (1 + (latitudeEffect * environmentalFactors.latitudinalEffect * 0.02));
    }
    
    // Limit pressure to realistic values
    pressurePredict = Math.max(880, Math.min(1020, pressurePredict));
    
    // Ensure wind speed and pressure are physically consistent
    // Lower pressure should correlate with higher wind speed
    const pressureWindCorrelation = this.correlateWindAndPressure(windPredict, pressurePredict);
    windPredict = pressureWindCorrelation.windSpeed;
    pressurePredict = pressureWindCorrelation.pressure;
    
    return {
      position: { lat: latPredict, lon: lonPredict },
      windSpeed: windPredict,
      pressure: pressurePredict
    };
  }
  
  /**
   * Determine the current phase of the storm based on history
   */
  determineStormPhase(history) {
    const historyLength = history.length;
    if (historyLength < 4) return 'early';
    if (historyLength > 12) return 'late';
    return 'peak';
  }
  
  /**
   * Adjust weights based on storm phase
   */
  adjustWeightsForPhase(weights, phase) {
    const phaseAdjustment = this.phaseAdjustments[phase] || this.phaseAdjustments.peak;
    const adjustedWeights = { ...weights };
    
    // Adjust position weights
    if (phaseAdjustment.position && phaseAdjustment.position.factor) {
      const factor = phaseAdjustment.position.factor;
      adjustedWeights.position = {
        lat: weights.position.lat.map(w => w * factor),
        lon: weights.position.lon.map(w => w * factor)
      };
    }
    
    // Adjust intensity weights
    if (phaseAdjustment.intensity && phaseAdjustment.intensity.factor) {
      const factor = phaseAdjustment.intensity.factor;
      adjustedWeights.intensity = weights.intensity.map(w => w * factor);
    }
    
    return adjustedWeights;
  }
  
  /**
   * Calculate the effect of sea surface temperature on hurricane intensification
   */
  calculateSSTEffect(sst) {
    // Hurricanes typically intensify over waters warmer than 26°C
    // and intensify rapidly over waters warmer than 28°C
    if (sst < 26) return -0.05; // Slight weakening effect
    if (sst < 28) return 0.02;  // Slight intensification
    if (sst < 30) return 0.05;  // Moderate intensification
    return 0.08;                // Strong intensification
  }
  
  /**
   * Calculate the effect of latitude on hurricane behavior
   */
  calculateLatitudeEffect(latitude, basin) {
    // Normalize latitude effect based on basin
    // Different basins have different latitude thresholds
    const absLat = Math.abs(latitude);
    
    let latThreshold;
    switch (basin) {
      case 'WP': // Western Pacific typhoons can sustain at higher latitudes
        latThreshold = 35;
        break;
      case 'NA': // North Atlantic
        latThreshold = 30;
        break;
      default:
        latThreshold = 28;
    }
    
    // No effect near equator
    if (absLat < 15) return 0;
    
    // Increasing effect as latitude increases
    if (absLat < latThreshold) {
      return (absLat - 15) / (latThreshold - 15) * 0.5;
    }
    
    // Strong effect beyond threshold
    return 0.5 + ((absLat - latThreshold) / 10) * 0.5;
  }
  
  /**
   * Ensure wind speed and pressure are physically consistent
   */
  correlateWindAndPressure(windSpeed, pressure) {
    // Use a simplified version of the wind-pressure relationship
    // P = 1010 - (windSpeed/1.15)^2/100
    // where windSpeed is in mph and pressure is in millibars
    
    // Calculate "ideal" pressure from wind
    const idealPressure = 1010 - Math.pow(windSpeed/1.15, 2)/100;
    
    // Calculate "ideal" wind from pressure
    const idealWind = 1.15 * Math.sqrt((1010 - pressure) * 100);
    
    // Average the current prediction with the "ideal" values
    return {
      windSpeed: (windSpeed + idealWind) / 2,
      pressure: (pressure + idealPressure) / 2
    };
  }
  
  /**
   * Combine predictions from ensemble members
   */
  combineEnsemblePredictions(predictions, state) {
    // Calculate weighted average of predictions
    // More weight is given to the main model
    
    // Simple arithmetic mean for position
    const totalLat = predictions.reduce((sum, pred) => sum + pred.position.lat, 0);
    const totalLon = predictions.reduce((sum, pred) => sum + pred.position.lon, 0);
    const avgLat = totalLat / predictions.length;
    const avgLon = totalLon / predictions.length;
    
    // Simple arithmetic mean for intensity
    const totalWind = predictions.reduce((sum, pred) => sum + pred.windSpeed, 0);
    const totalPressure = predictions.reduce((sum, pred) => sum + pred.pressure, 0);
    const avgWind = totalWind / predictions.length;
    const avgPressure = totalPressure / predictions.length;
    
    // Calculate standard deviations for uncertainty estimation
    const latStdDev = this.calculateStdDev(predictions.map(p => p.position.lat), avgLat);
    const lonStdDev = this.calculateStdDev(predictions.map(p => p.position.lon), avgLon);
    const windStdDev = this.calculateStdDev(predictions.map(p => p.windSpeed), avgWind);
    const pressureStdDev = this.calculateStdDev(predictions.map(p => p.pressure), avgPressure);
    
    return {
      position: { lat: avgLat, lon: avgLon },
      windSpeed: avgWind,
      pressure: avgPressure,
      uncertainty: {
        position: { lat: latStdDev, lon: lonStdDev },
        windSpeed: windStdDev,
        pressure: pressureStdDev
      }
    };
  }
  
  /**
   * Calculate standard deviation
   */
  calculateStdDev(values, mean) {
    const squareDiffs = values.map(value => Math.pow(value - mean, 2));
    const avgSquareDiff = squareDiffs.reduce((sum, value) => sum + value, 0) / values.length;
    return Math.sqrt(avgSquareDiff);
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
      this.weights.position.lat[i] += this.learningRate * latError * reward / 100;
      this.weights.position.lon[i] += this.learningRate * lonError * reward / 100;
    }
    
    // Update intensity weights
    const intensityError = actual.windSpeed - prediction.windSpeed;
    for (let i = 0; i < this.weights.intensity.length; i++) {
      this.weights.intensity[i] += this.learningRate * intensityError * reward / 100;
    }
    
    // Update pressure weights
    const pressureError = actual.pressure - prediction.pressure;
    for (let i = 0; i < this.weights.pressure.length; i++) {
      this.weights.pressure[i] += this.learningRate * pressureError * reward / 100;
    }
    
    // If using basin-specific models, update those too
    if (this.options.useBasinModels && actual.basin) {
      const basin = actual.basin;
      if (this.basinModels[basin]) {
        for (let i = 0; i < this.basinModels[basin].position.lat.length; i++) {
          this.basinModels[basin].position.lat[i] += this.learningRate * latError * reward / 100 * 1.5; // Higher learning rate for basin-specific
          this.basinModels[basin].position.lon[i] += this.learningRate * lonError * reward / 100 * 1.5;
        }
        
        for (let i = 0; i < this.basinModels[basin].intensity.length; i++) {
          this.basinModels[basin].intensity[i] += this.learningRate * intensityError * reward / 100 * 1.5;
        }
        
        for (let i = 0; i < this.basinModels[basin].pressure.length; i++) {
          this.basinModels[basin].pressure[i] += this.learningRate * pressureError * reward / 100 * 1.5;
        }
      }
    }
    
    // Update ensemble members (with smaller updates)
    this.ensembleMembers.forEach(member => {
      for (let i = 0; i < member.weights.position.lat.length; i++) {
        member.weights.position.lat[i] += member.learningRate * latError * reward / 100 * 0.5;
        member.weights.position.lon[i] += member.learningRate * lonError * reward / 100 * 0.5;
      }
      
      for (let i = 0; i < member.weights.intensity.length; i++) {
        member.weights.intensity[i] += member.learningRate * intensityError * reward / 100 * 0.5;
      }
      
      for (let i = 0; i < member.weights.pressure.length; i++) {
        member.weights.pressure[i] += member.learningRate * pressureError * reward / 100 * 0.5;
      }
    });
  }

  /**
   * Train the agent on historical data
   */
  async train(environment, episodes = 100) {
    console.log(`Training hurricane prediction agent on ${episodes} episodes...`);
    const performance = [];
    
    for (let episode = 0; episode < episodes; episode++) {
      // Reset the environment and get initial state
      const state = environment.reset();
      
      let done = false;
      const history = [];
      
      // Episode loop
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
      
      // Log training progress
      if ((episode + 1) % 5 === 0 || episode === 0 || episode === episodes - 1) {
        console.log(`Episode ${episode + 1}/${episodes} - ` + 
          `Avg Position Error: ${episodePerformance.avgPosError.toFixed(2)} km, ` + 
          `Avg Intensity Error: ${episodePerformance.avgIntensityError.toFixed(2)} mph`);
      }
    }
    
    // Store performance history
    this.trainingPerformance = performance;
    
    return performance;
  }
  
  /**
   * Create a slightly different version of this agent for ensemble forecasting
   */
  createVariation(variationFactor = 0.3) {
    const variedAgent = new HurricanePredictionAgent({
      ...this.options,
      ensembleSize: Math.max(1, this.options.ensembleSize - 1) // Smaller ensemble to save computation
    });
    
    // Vary position weights
    for (let i = 0; i < this.weights.position.lat.length; i++) {
      variedAgent.weights.position.lat[i] = this.weights.position.lat[i] * 
        (1 + (Math.random() * variationFactor * 2 - variationFactor));
      variedAgent.weights.position.lon[i] = this.weights.position.lon[i] * 
        (1 + (Math.random() * variationFactor * 2 - variationFactor));
    }
    
    // Vary intensity weights
    for (let i = 0; i < this.weights.intensity.length; i++) {
      variedAgent.weights.intensity[i] = this.weights.intensity[i] * 
        (1 + (Math.random() * variationFactor * 2 - variationFactor));
    }
    
    // Vary pressure weights
    for (let i = 0; i < this.weights.pressure.length; i++) {
      variedAgent.weights.pressure[i] = this.weights.pressure[i] * 
        (1 + (Math.random() * variationFactor * 2 - variationFactor));
    }
    
    // Vary environmental factors
    Object.keys(this.environmentalFactors).forEach(factor => {
      variedAgent.environmentalFactors[factor] = this.environmentalFactors[factor] * 
        (1 + (Math.random() * variationFactor - variationFactor/2));
    });
    
    // Vary learning rate
    variedAgent.learningRate = this.learningRate * 
      (1 + (Math.random() * 0.4 - 0.2));
    
    return variedAgent;
  }

  /**
   * Get forecast statistics including uncertainty
   */
  getForecastStatistics(predictions) {
    if (!predictions || predictions.length === 0) {
      return null;
    }
    
    // Calculate confidence intervals for each forecast time
    const statistics = {};
    
    // Group predictions by forecast day
    const forecastDays = [...new Set(predictions.map(p => p.day))].sort();
    
    forecastDays.forEach(day => {
      const dayPredictions = predictions.filter(p => p.day === day);
      
      if (dayPredictions.length === 0) return;
      
      // Calculate position statistics
      const lats = dayPredictions.map(p => p.position.lat);
      const lons = dayPredictions.map(p => p.position.lon);
      const avgLat = lats.reduce((sum, lat) => sum + lat, 0) / lats.length;
      const avgLon = lons.reduce((sum, lon) => sum + lon, 0) / lons.length;
      const latStdDev = this.calculateStdDev(lats, avgLat);
      const lonStdDev = this.calculateStdDev(lons, avgLon);
      
      // Calculate intensity statistics
      const windSpeeds = dayPredictions.map(p => p.windSpeed);
      const avgWindSpeed = windSpeeds.reduce((sum, ws) => sum + ws, 0) / windSpeeds.length;
      const windStdDev = this.calculateStdDev(windSpeeds, avgWindSpeed);
      
      // Calculate category ranges
      const categories = dayPredictions.map(p => 
        typeof p.category === 'string' ? parseInt(p.category) || 0 : p.category || 0
      );
      const minCategory = Math.min(...categories);
      const maxCategory = Math.max(...categories);
      
      // Calculate confidence based on spread
      // Lower spread = higher confidence
      const positionSpread = Math.sqrt(latStdDev * latStdDev + lonStdDev * lonStdDev);
      const intensitySpread = windStdDev;
      
      // Normalize spreads (lower is better)
      const normalizedPositionSpread = Math.min(1, positionSpread / 2); // 2 degrees is max for normalization
      const normalizedIntensitySpread = Math.min(1, intensitySpread / 40); // 40 mph is max for normalization
      
      // Calculate confidence (100% - spread)
      const positionConfidence = 100 * (1 - normalizedPositionSpread);
      const intensityConfidence = 100 * (1 - normalizedIntensitySpread);
      
      // Average confidence
      const overallConfidence = (positionConfidence + intensityConfidence) / 2;
      
      statistics[day] = {
        position: {
          mean: { lat: avgLat, lon: avgLon },
          stdDev: { lat: latStdDev, lon: lonStdDev },
          confidence: positionConfidence
        },
        intensity: {
          mean: avgWindSpeed,
          stdDev: windStdDev,
          range: [avgWindSpeed - 2*windStdDev, avgWindSpeed + 2*windStdDev],
          confidence: intensityConfidence
        },
        category: {
          range: [minCategory, maxCategory],
          mode: this.getMode(categories)
        },
        confidence: overallConfidence
      };
    });
    
    return statistics;
  }
  
  /**
   * Get the most frequent value in an array
   */
  getMode(values) {
    const counts = {};
    let maxCount = 0;
    let mode = null;
    
    values.forEach(value => {
      counts[value] = (counts[value] || 0) + 1;
      if (counts[value] > maxCount) {
        maxCount = counts[value];
        mode = value;
      }
    });
    
    return mode;
  }
}

export { HurricaneEnvironment, HurricanePredictionAgent };