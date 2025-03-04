// hurricaneData.js

/**
 * Service to collect and preprocess historical hurricane data for agent training
 */

/**
 * Fetch historical hurricane tracks from the National Hurricane Center archive
 */
async function fetchHistoricalHurricaneData() {
    try {
      // In a real implementation, this would fetch from NOAA's historical hurricane database
      // For now, we'll use a simulated dataset
      return simulateHistoricalData();
    } catch (error) {
      console.error('Error fetching historical hurricane data:', error);
      throw error;
    }
  }
  
  /**
   * Simulate historical hurricane data for development
   */
  function simulateHistoricalData() {
    const hurricanes = [];
    const years = 10; // Generate data for 10 years
    const hurricanesPerYear = 5; // Average 5 hurricanes per year
    
    for (let year = 2015; year < 2015 + years; year++) {
      for (let i = 0; i < hurricanesPerYear; i++) {
        const id = `H${year}${i+1}`;
        const name = `Hurricane ${String.fromCharCode(65 + i)}`;
        const duration = 5 + Math.floor(Math.random() * 10); // 5-14 days
        const startTime = new Date(year, 5 + Math.floor(Math.random() * 4), Math.floor(Math.random() * 28) + 1);
        
        // Generate track data
        const track = generateHurricaneTrack(duration);
        
        hurricanes.push({
          id,
          name,
          year,
          startTime,
          initialPosition: track[0].position,
          initialWindSpeed: track[0].windSpeed,
          initialPressure: track[0].pressure,
          track
        });
      }
    }
    
    return hurricanes;
  }
  
  /**
   * Generate a realistic hurricane track
   */
  function generateHurricaneTrack(days) {
    const track = [];
    const timeSteps = days * 4; // 4 observations per day (every 6 hours)
    
    // Starting position - random location in hurricane formation regions
    const regions = [
      { lat: 15, lon: -60 },  // Caribbean
      { lat: 20, lon: -45 },  // Central Atlantic
      { lat: 12, lon: -35 }   // East Atlantic
    ];
    
    const region = regions[Math.floor(Math.random() * regions.length)];
    let lat = region.lat + (Math.random() * 4) - 2;
    let lon = region.lon + (Math.random() * 4) - 2;
    
    // Initial intensity
    let windSpeed = 30 + Math.random() * 20; // 30-50 mph
    let pressure = 1010 - (windSpeed / 2); // Roughly correlate with wind speed
    
    // Typical hurricane movement is NW in Atlantic
    let latSpeed = 0.1 + Math.random() * 0.2; // 0.1-0.3 degrees per time step
    let lonSpeed = -0.2 - Math.random() * 0.3; // -0.2 to -0.5 degrees per time step
    
    for (let i = 0; i < timeSteps; i++) {
      const timestamp = new Date();
      timestamp.setHours(timestamp.getHours() + (i * 6));
      
      // Add some random variation to movement
      latSpeed += (Math.random() * 0.1) - 0.05;
      lonSpeed += (Math.random() * 0.1) - 0.05;
      
      // Constrain speeds to realistic values
      latSpeed = Math.max(-0.3, Math.min(0.5, latSpeed));
      lonSpeed = Math.max(-0.6, Math.min(0.1, lonSpeed));
      
      // Update position
      lat += latSpeed;
      lon += lonSpeed;
      
      // Update intensity based on typical lifecycle
      // Intensification in first third, peak in middle, weakening in final third
      if (i < timeSteps / 3) {
        windSpeed += (2 + Math.random() * 5); // Intensification
      } else if (i > (timeSteps * 2) / 3) {
        windSpeed -= (1 + Math.random() * 6); // Weakening
      } else {
        windSpeed += (Math.random() * 8) - 4; // Fluctuation near peak
      }
      
      // Limit wind speed to realistic values
      windSpeed = Math.max(30, Math.min(175, windSpeed));
      
      // Pressure is inversely related to wind speed
      pressure = 1010 - (windSpeed / 2);
      
      track.push({
        position: { lat, lon },
        windSpeed,
        pressure,
        timestamp
      });
    }
    
    return track;
  }
  
  /**
   * Preprocess data for agent training
   */
  function preprocessDataForTraining(hurricanes) {
    // Extract features and normalize
    return hurricanes.map(hurricane => {
      return {
        ...hurricane,
        track: hurricane.track.map(point => ({
          ...point,
          // Add derived features
          windSpeedChange: point.windSpeed - hurricane.initialWindSpeed,
          pressureChange: point.pressure - hurricane.initialPressure,
          // Could add more features here like:
          // - Distance from initial position
          // - Time since formation
          // - Relative position to coastlines
        }))
      };
    });
  }
  
  export { fetchHistoricalHurricaneData, preprocessDataForTraining };