// hurricaneAgent.js - Simplified version that redirects to Python API

/**
 * This file is maintained for compatibility with existing code
 * The actual implementation has been moved to Python
 */

class HurricaneEnvironment {
  constructor() {
    console.log("Warning: Using stub HurricaneEnvironment - actual implementation is in Python backend");
  }
  
  // Stub methods to maintain compatibility
  async initialise() { return {}; }
  reset() { return {}; }
  getState() { return {}; }
  step() { return { state: {}, reward: 0, done: true }; }
  getActualForTimeStep() { return null; }
  evaluatePerformance() { return { avgPosError: 0, avgIntensityError: 0 }; }
}

class HurricanePredictionAgent {
  constructor() {
    console.log("Warning: Using stub HurricanePredictionAgent - actual implementation is in Python backend");
  }
  
  // Stub methods to maintain compatibility
  predict() { return {}; }
  update() {}
  async train() { return []; }
  createVariation() { return new HurricanePredictionAgent(); }
  getForecastStatistics() { return {}; }
}

export { HurricaneEnvironment, HurricanePredictionAgent };