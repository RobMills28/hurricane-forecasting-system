// noaaService.js

/**
 * Service to handle NOAA API interactions
 * Documentation: https://www.weather.gov/documentation/services-web-api
 */

const BASE_URL = 'https://api.weather.gov';
const USER_AGENT = 'Atlas Command Center (robmills2000@hotmail.com)';

/**
 * Fetches active severe weather events including hurricanes
 */
export async function getActiveHurricanes() {
  try {
    // Get all active alerts
    const response = await fetch(`${BASE_URL}/alerts/active`, {
      headers: {
        'User-Agent': USER_AGENT,
        'Accept': 'application/geo+json'
      }
    });

    if (!response.ok) throw new Error('Failed to fetch weather data');
    
    const data = await response.json();
    
    // Filter for relevant severe weather events
    const severeWeatherAlerts = data.features.filter(feature => {
      const event = feature.properties.event.toLowerCase();
      return event.includes('hurricane') || 
             event.includes('tropical') ||
             event.includes('flood') ||
             event.includes('severe') ||
             event.includes('storm');
    });

    return processWeatherAlerts(severeWeatherAlerts);
  } catch (error) {
    console.error('Error fetching weather data:', error);
    throw error;
  }
}

/**
 * Process weather alerts into a standardised format
 */
function processWeatherAlerts(alerts) {
  return alerts.map(alert => {
    const properties = alert.properties;
    const geometry = alert.geometry;

    return {
      id: properties.id,
      type: properties.event,
      severity: properties.severity,
      certainty: properties.certainty,
      urgency: properties.urgency,
      name: getEventName(properties),
      description: properties.description,
      instruction: properties.instruction,
      areas: properties.areaDesc,
      coordinates: extractCoordinates(geometry),
      status: properties.status,
      messageType: properties.messageType,
      category: getEventCategory(properties),
      onset: properties.onset,
      expires: properties.expires,
      headline: properties.headline
    };
  });
}

/**
 * Extracts event name from properties
 */
function getEventName(properties) {
  // First check if it's a hurricane/tropical storm
  const stormMatch = properties.headline?.match(/Hurricane\s+(\w+)|Tropical\s+Storm\s+(\w+)/i);
  if (stormMatch) {
    return stormMatch[1] || stormMatch[2];
  }

  // For other events, use the event type and location
  return `${properties.event} - ${properties.areaDesc.split(';')[0]}`;
}

/**
 * Determines event category/severity
 */
function getEventCategory(properties) {
  // Check for hurricane category first
  const categoryMatch = properties.description?.match(/Category\s+(\d)/i);
  if (categoryMatch) {
    return parseInt(categoryMatch[1]);
  }

  // For other events, map severity to a numeric scale
  switch (properties.severity.toLowerCase()) {
    case 'extreme':
      return 5;
    case 'severe':
      return 4;
    case 'moderate':
      return 3;
    case 'minor':
      return 2;
    default:
      return 1;
  }
}

/**
 * Extracts coordinates from geometry
 */
function extractCoordinates(geometry) {
  if (!geometry) return null;
  
  switch (geometry.type) {
    case 'Point':
      return geometry.coordinates;
    case 'Polygon':
      return getCenterOfPolygon(geometry.coordinates[0]);
    case 'MultiPolygon':
      // Take the center of the first polygon for simplicity
      return getCenterOfPolygon(geometry.coordinates[0][0]);
    default:
      return null;
  }
}

/**
 * Calculates center point of a polygon
 */
function getCenterOfPolygon(coordinates) {
  const count = coordinates.length;
  const center = coordinates.reduce(
    (acc, coord) => [acc[0] + coord[0], acc[1] + coord[1]],
    [0, 0]
  );
  return [center[0] / count, center[1] / count];
}

/**
 * Fetches specific zone forecast data
 */
export async function getZoneForecast(zoneId) {
  try {
    const response = await fetch(`${BASE_URL}/zones/forecast/${zoneId}/forecast`, {
      headers: {
        'User-Agent': USER_AGENT,
        'Accept': 'application/geo+json'
      }
    });

    if (!response.ok) throw new Error('Failed to fetch zone forecast');
    
    const data = await response.json();
    return data.properties.periods;
  } catch (error) {
    console.error('Error fetching zone forecast:', error);
    throw error;
  }
}

/**
 * Fetches detailed weather observations for a specific point
 */
export async function getHurricaneObservations(lat, lon) {
  try {
    // First, try getting the nearest station - with better error handling
    const stationResponse = await fetch(
      `${BASE_URL}/points/${lat},${lon}/stations`, {
        headers: {
          'User-Agent': USER_AGENT,
          'Accept': 'application/geo+json'
        },
        // Add timeout to prevent hanging requests
        signal: AbortSignal.timeout(10000) // 10 second timeout
      }
    );

    if (!stationResponse.ok) {
      console.warn(`Station fetch failed: ${stationResponse.status} ${stationResponse.statusText}`);
      
      // Try an alternative endpoint or approach if the main one fails
      // If lat/lon is in US, try a secondary endpoint or static list of reliable stations
      if (isInUnitedStates(lat, lon)) {
        // For US locations, I could try a hardcoded reliable station nearby
        return await tryBackupUSStation(lat, lon);
      } else {
        // If outside US, try OpenMeteo as an alternative source
        // This will maintain consistent global coverage
        return await getOpenMeteoObservations(lat, lon, determineRegionFromCoords(lat, lon));
      }
    }
    
    const stationData = await stationResponse.json();
    
    // If no stations found, use alternative approach
    if (!stationData.features || stationData.features.length === 0) {
      console.warn('No weather stations found nearby');
      return await getOpenMeteoObservations(lat, lon, determineRegionFromCoords(lat, lon));
    }

    // Try to get the closest station
    const stationId = stationData.features[0].id;
    
    // Attempt to get observations from the station
    const obsResponse = await fetch(
      `${stationId}/observations/latest`, {
        headers: {
          'User-Agent': USER_AGENT,
          'Accept': 'application/geo+json'
        },
        // Add timeout to prevent hanging requests
        signal: AbortSignal.timeout(10000) // 10 second timeout
      }
    );

    if (!obsResponse.ok) {
      console.warn(`Failed to fetch observations: ${obsResponse.status} ${obsResponse.statusText}`);
      // If station data fetch fails, fall back to OpenMeteo for this location
      return await getOpenMeteoObservations(lat, lon, determineRegionFromCoords(lat, lon));
    }
    
    const obsData = await obsResponse.json();
    
    // Process and return the observation data
    return processObservations(obsData.properties);
  } catch (error) {
    console.error('Error in getHurricaneObservations:', error);
    // Last resort fallback
    return await getOpenMeteoObservations(lat, lon, determineRegionFromCoords(lat, lon));
  }
}

/**
 * Determine if coordinates are in the United States
 */
function isInUnitedStates(lat, lon) {
  // Simple bounding box for continental US
  return (lat > 24.0 && lat < 50.0 && lon > -125.0 && lon < -66.0);
}

/**
 * Determine region code based on coordinates
 */
function determineRegionFromCoords(lat, lon) {
  // Simple region determination based on coordinates
  if (lon > -100 && lon < 0 && lat > 0) return 'NA'; // North Atlantic
  if (lon >= -180 && lon < -100 && lat > 0) return 'EP'; // Eastern Pacific
  if (lon >= 100 && lon < 180 && lat > 0) return 'WP'; // Western Pacific
  if (lon >= 40 && lon < 100 && lat > 0) return 'NI'; // North Indian
  if (lon >= 40 && lon < 135 && lat <= 0) return 'SI'; // South Indian
  if (lon >= 135 && lon < 180 && lat <= 0) return 'SP'; // South Pacific
  
  return 'GLOBAL'; // Default
}

/**
 * Try using a backup station for US locations
 */
async function tryBackupUSStation(lat, lon) {
  // Known reliable stations by region (example)
  const backupStations = {
    east: 'https://api.weather.gov/stations/KBOS/observations/latest',
    west: 'https://api.weather.gov/stations/KSFO/observations/latest',
    central: 'https://api.weather.gov/stations/KORD/observations/latest',
    south: 'https://api.weather.gov/stations/KMIA/observations/latest'
  };
  
  // Determine which region the coordinates are in
  let region;
  if (lon < -100) region = 'west';
  else if (lon < -87) region = 'central';
  else if (lat < 36) region = 'south';
  else region = 'east';
  
  try {
    // Try the backup station
    const response = await fetch(backupStations[region], {
      headers: {
        'User-Agent': USER_AGENT,
        'Accept': 'application/geo+json'
      }
    });
    
    if (!response.ok) {
      // If this fails too, fall back to OpenMeteo
      return await getOpenMeteoObservations(lat, lon, determineRegionFromCoords(lat, lon));
    }
    
    const data = await response.json();
    return processObservations(data.properties);
  } catch (error) {
    // If all else fails, use OpenMeteo
    return await getOpenMeteoObservations(lat, lon, determineRegionFromCoords(lat, lon));
  }
}

/**
 * Process weather observations into a standardised format
 */
function processObservations(obs) {
  // Handle potentially null values with default values
  return {
    timestamp: obs.timestamp,
    temperature: obs.temperature?.value ?? null,
    windSpeed: obs.windSpeed?.value ?? null,
    windDirection: obs.windDirection?.value ?? null,
    barometricPressure: obs.barometricPressure?.value ?? null,
    seaLevelPressure: obs.seaLevelPressure?.value ?? null,
    visibility: obs.visibility?.value ?? null,
    relativeHumidity: obs.relativeHumidity?.value ?? null,
    windGust: obs.windGust?.value ?? null,
    precipitationLastHour: obs.precipitationLastHour?.value ?? null
  };
}

/**
 * Get color coding for different event types
 */
export function getEventColor(event) {
  const eventType = event.toLowerCase();
  if (eventType.includes('hurricane')) return '#ff4500';
  if (eventType.includes('tropical storm')) return '#ffd700';
  if (eventType.includes('flood')) return '#4169e1';
  if (eventType.includes('severe')) return '#ff0000';
  if (eventType.includes('storm')) return '#9932cc';
  return '#ffffff';
}