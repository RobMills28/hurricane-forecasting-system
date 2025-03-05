// hurricaneData.js

/**
 * Service to collect and preprocess historical hurricane data for agent training
 */
import { fetchIBTraCSHurricanes, getBasinStats } from './ibtracsConnector';

// Cache for processed data to avoid repeated fetches
let cachedHurricanes = null;
let dataLoadPromise = null;

/**
 * Fetch historical hurricane tracks from IBTrACS
 */
export async function fetchHistoricalHurricaneData(options = {}) {
  try {
    // If we're already loading data, return the existing promise
    if (dataLoadPromise) {
      return dataLoadPromise;
    }
    
    // If we have cached data and aren't forced to reload, use cache
    if (cachedHurricanes && !options.forceReload) {
      console.log('Using cached hurricane data');
      return cachedHurricanes;
    }
    
    // Set default options for hurricane data
    const defaultOptions = {
      fullHistory: false,  // Use last3years by default for faster loading
      minTrackPoints: 5,   // Ensure storms have enough data points
      minCategory: 0       // Include all storms by default
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    console.log('Fetching historical hurricane data with options:', mergedOptions);
    
    // Create a promise and cache it
    dataLoadPromise = fetchIBTraCSHurricanes(mergedOptions)
      .then(data => {
        console.log(`Loaded ${data.length} historical hurricanes`);
        // Log basin statistics
        const basinStats = getBasinStats(data);
        console.log('Basin statistics:', basinStats.counts);
        
        // Cache the data
        cachedHurricanes = data;
        // Clear the loading promise
        dataLoadPromise = null;
        return data;
      })
      .catch(error => {
        console.error('Error loading hurricane data:', error);
        // Clear the loading promise on error
        dataLoadPromise = null;
        // Fall back to simulated data if real data fails
        console.warn('Falling back to simulated data');
        return simulateHistoricalData();
      });
    
    return dataLoadPromise;
  } catch (error) {
    console.error('Error in fetchHistoricalHurricaneData:', error);
    return simulateHistoricalData();
  }
}

/**
 * Simulate historical hurricane data for development (fallback)
 * This is only used if real data loading fails
 */
function simulateHistoricalData() {
  console.warn('Using simulated hurricane data');
  const hurricanes = [];
  const years = 10;
  const hurricanesPerYear = 5;
  
  for (let year = 2015; year < 2015 + years; year++) {
    for (let i = 0; i < hurricanesPerYear; i++) {
      const id = `H${year}${i+1}`;
      const name = `Hurricane ${String.fromCharCode(65 + i)}`;
      const duration = 5 + Math.floor(Math.random() * 10);
      const startTime = new Date(year, 5 + Math.floor(Math.random() * 4), Math.floor(Math.random() * 28) + 1);
      
      // Generate track data
      const track = generateHurricaneTrack(duration);
      
      hurricanes.push({
        id,
        name,
        year,
        basin: 'NA', // North Atlantic
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
 * Generate a simulated hurricane track (fallback)
 */
function generateHurricaneTrack(days) {
  // Implementation unchanged from your original code
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
export function preprocessDataForTraining(hurricanes) {
  // Extract features and normalize
  return hurricanes.map(hurricane => {
    // Calculate the maximum wind speed for the hurricane
    const maxWindSpeed = Math.max(...hurricane.track.map(point => point.windSpeed || 0));
    
    return {
      ...hurricane,
      maxWindSpeed,
      track: hurricane.track.map((point, index, track) => {
        // Skip first point for derivatives
        const prevPoint = index > 0 ? track[index - 1] : point;
        
        // Calculate time difference in hours
        const timeDiff = (point.timestamp - prevPoint.timestamp) / (1000 * 60 * 60);
        
        // Position change rate (degrees per hour)
        const latChangeRate = timeDiff > 0 ? 
          (point.position.lat - prevPoint.position.lat) / timeDiff : 0;
        
        const lonChangeRate = timeDiff > 0 ? 
          (point.position.lon - prevPoint.position.lon) / timeDiff : 0;
        
        // Intensity change rate (mph per hour)
        const windSpeedChangeRate = timeDiff > 0 ? 
          (point.windSpeed - prevPoint.windSpeed) / timeDiff : 0;
        
        // Pressure change rate (mb per hour)
        const pressureChangeRate = timeDiff > 0 ?
          (point.pressure - prevPoint.pressure) / timeDiff : 0;
        
        // Normalize wind speed relative to maximum
        const normalizedWindSpeed = point.windSpeed / maxWindSpeed;
        
        // Calculate distance from genesis (initial position)
        const distanceFromGenesis = calculateDistance(
          hurricane.initialPosition,
          point.position
        );
        
        // Time since genesis (in hours)
        const hoursSinceGenesis = (point.timestamp - hurricane.startTime) / (1000 * 60 * 60);
        
        return {
          ...point,
          // Original derived features
          windSpeedChange: point.windSpeed - hurricane.initialWindSpeed,
          pressureChange: point.pressure - hurricane.initialPressure,
          
          // New derived features
          latChangeRate,
          lonChangeRate,
          windSpeedChangeRate,
          pressureChangeRate,
          normalizedWindSpeed,
          distanceFromGenesis,
          hoursSinceGenesis,
          
          // Phase of storm lifecycle (early, peak, late)
          stormPhase: getStormPhase(index, track.length)
        };
      })
    };
  });
}

/**
 * Get the phase of a storm based on position in track
 * @param {number} index - Current index
 * @param {number} totalLength - Total track length
 * @returns {string} - Storm phase
 */
function getStormPhase(index, totalLength) {
  const normalizedPosition = index / totalLength;
  
  if (normalizedPosition < 0.3) return 'early';    // Formation/Intensification
  if (normalizedPosition < 0.7) return 'peak';     // Mature/Peak
  return 'late';                                   // Weakening/Dissipation
}

/**
 * Calculate distance between two geographic points (in km)
 */
function calculateDistance(pos1, pos2) {
  if (!pos1 || !pos2) return 0;
  
  // Haversine formula for calculating distance between two points on Earth
  const R = 6371; // Earth's radius in km
  const dLat = deg2rad(pos2.lat - pos1.lat);
  const dLon = deg2rad(pos2.lon - pos1.lon);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(deg2rad(pos1.lat)) * Math.cos(deg2rad(pos2.lat)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function deg2rad(deg) {
  return deg * (Math.PI / 180);
}