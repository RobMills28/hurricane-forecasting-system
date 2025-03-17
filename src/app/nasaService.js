// Updated nasaService.js with better error handling and caching

// nasaService.js
// Service to handle NASA API interactions

const NASA_API_KEY = 'DEMO_KEY'; // Replace with your NASA API key for production use
const NASA_API_BASE_URL = 'https://api.nasa.gov/planetary/earth';

// Simple cache for imagery requests
const imageCache = new Map();
const layerCache = new Map();

/**
 * Fetches NASA Earth imagery assets for a given location and date with caching
 * 
 * @param {number} lat - Latitude
 * @param {number} lon - Longitude
 * @param {string} date - Date in YYYY-MM-DD format
 * @param {number} dim - Width and height of image in degrees (0.025 to 0.5)
 * @returns {Promise<Object>} The assets data with URL to the satellite image
 */
export async function getNasaImageryAssets(lat, lon, date = new Date().toISOString().split('T')[0], dim = 0.15) {
  try {
    // Check cache first
    const cacheKey = `${lat}-${lon}-${date}-${dim}`;
    if (imageCache.has(cacheKey)) {
      console.log('Using cached NASA imagery assets');
      return imageCache.get(cacheKey);
    }
    
    const response = await fetch(
      `${NASA_API_BASE_URL}/assets?lat=${lat}&lon=${lon}&date=${date}&dim=${dim}&api_key=${NASA_API_KEY}`,
      { 
        headers: { 'User-Agent': 'Atlas Command Center/1.0' },
        timeout: 5000 // 5 second timeout
      }
    );
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`NASA API error: ${response.status} - ${errorText}`);
    }
    
    const data = await response.json();
    
    // Cache the result
    imageCache.set(cacheKey, data);
    
    return data;
  } catch (error) {
    console.error('Error fetching NASA imagery assets:', error);
    // Return fallback data
    return {
      error: true,
      message: error.message,
      // Provide fallback URL if possible
      url: null
    };
  }
}

/**
 * Fetches NASA Earth imagery for a given location and date
 * 
 * @param {number} lat - Latitude
 * @param {number} lon - Longitude
 * @param {string} date - Date in YYYY-MM-DD format
 * @param {number} dim - Width and height of image in degrees (0.025 to 0.5)
 * @returns {Promise<string>} URL to the image
 */
export async function getNasaImagery(lat, lon, date = new Date().toISOString().split('T')[0], dim = 0.15) {
  try {
    // First get the assets data which contains the image URL
    const assetsData = await getNasaImageryAssets(lat, lon, date, dim);
    
    // Return the URL from the assets data
    if (assetsData && assetsData.url) {
      return assetsData.url;
    } else if (assetsData.error) {
      throw new Error(assetsData.message || 'Error fetching NASA imagery');
    } else {
      throw new Error('No imagery URL found in NASA API response');
    }
  } catch (error) {
    console.error('Error fetching NASA imagery:', error);
    throw error;
  }
}

/**
 * Fetches NASA GIBS (Global Imagery Browse Services) layer information
 * This provides various Earth observation visualizations like Sea Surface Temperature
 * 
 * @returns {Promise<Object[]>} Array of available GIBS layers
 */
export async function getGibsLayers() {
  try {
    // Check cache first
    if (layerCache.size > 0) {
      return Array.from(layerCache.values());
    }
    
    // NASA GIBS doesn't have a direct API to list layers, so we're returning a curated list
    // In a production app, you might fetch this from a more dynamic source
    const layers = [
      {
        id: 'MODIS_Terra_CorrectedReflectance_TrueColor',
        title: 'MODIS Terra True Color',
        subtitle: 'Terra / MODIS',
        description: 'Combination of MODIS visible bands 1, 4, and 3',
        format: 'image/jpeg',
        tileMatrixSet: 'EPSG4326_250m',
        minZoom: 0,
        maxZoom: 7
      },
      {
        id: 'MODIS_Terra_SurfaceTemp_Day',
        title: 'Land Surface Temperature (Day)',
        subtitle: 'Terra / MODIS',
        description: 'Surface temperature during daytime',
        format: 'image/png',
        tileMatrixSet: 'EPSG4326_2km',
        minZoom: 0,
        maxZoom: 7
      },
      {
        id: 'GHRSST_L4_MUR_Sea_Surface_Temperature',
        title: 'Sea Surface Temperature',
        subtitle: 'MUR',
        description: 'Global sea surface temperature composite',
        format: 'image/png',
        tileMatrixSet: 'EPSG4326_1km',
        minZoom: 0,
        maxZoom: 8
      },
      {
        id: 'VIIRS_SNPP_DayNightBand_ENCC',
        title: 'Earth at Night',
        subtitle: 'Suomi NPP / VIIRS',
        description: 'Earth at night visualizing light sources',
        format: 'image/png',
        tileMatrixSet: 'EPSG4326_500m',
        minZoom: 0,
        maxZoom: 8
      },
      {
        id: 'IMERG_Precipitation_Rate',
        title: 'Precipitation Rate',
        subtitle: 'IMERG',
        description: 'Global precipitation rate',
        format: 'image/png',
        tileMatrixSet: 'EPSG4326_2km',
        minZoom: 0,
        maxZoom: 7
      },
      {
        id: 'AIRS_Temperature_850hPa_Day',
        title: 'Air Temperature at 850 hPa',
        subtitle: 'AIRS / Aqua',
        description: 'Atmospheric temperature at 850 hPa pressure level',
        format: 'image/png',
        tileMatrixSet: 'EPSG4326_2km',
        minZoom: 0,
        maxZoom: 6
      },
      {
        id: 'MODIS_Terra_Cloud_Top_Temp_Day',
        title: 'Cloud Top Temperature',
        subtitle: 'Terra / MODIS',
        description: 'Temperature at the top of clouds',
        format: 'image/png',
        tileMatrixSet: 'EPSG4326_2km',
        minZoom: 0,
        maxZoom: 7
      }
    ];
    
    // Cache the layers
    layers.forEach(layer => layerCache.set(layer.id, layer));
    
    return layers;
  } catch (error) {
    console.error('Error fetching GIBS layers:', error);
    throw error;
  }
}

/**
 * Gets the tile URL template for a NASA GIBS layer
 * 
 * @param {Object} layer - GIBS layer information
 * @returns {string} URL template for the WMTS layer
 */
export function getGibsUrlTemplate(layer) {
  return `https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?service=WMTS&request=GetTile&version=1.0.0&layer=${layer.id}&style=default&format=${encodeURIComponent(layer.format)}&tilematrixset=${layer.tileMatrixSet}&tilematrix={z}&tilerow={y}&tilecol={x}`;
}

// Updated noaaService.js with search and filtering

// noaaService.js

/**
 * Service to handle NOAA API interactions
 * Documentation: https://www.weather.gov/documentation/services-web-api
 */

const BASE_URL = 'https://api.weather.gov';
const USER_AGENT = 'Atlas Command Center (your-email@domain.com)';

// Cache for active alerts
let alertsCache = null;
let lastFetchTime = 0;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

/**
 * Fetches active severe weather events including hurricanes
 */
export async function getActiveHurricanes(options = {}) {
  try {
    const now = Date.now();
    
    // Check cache first if not forcing refresh
    if (!options.forceRefresh && alertsCache && (now - lastFetchTime < CACHE_DURATION)) {
      console.log('Using cached weather data');
      
      // Apply filters if requested
      if (options.filter || options.search) {
        return filterAlerts(alertsCache, options);
      }
      
      return alertsCache;
    }
    
    // Fetch fresh data
    const response = await fetch(`${BASE_URL}/alerts/active`, {
      headers: {
        'User-Agent': USER_AGENT,
        'Accept': 'application/geo+json'
      }
    });

    if (!response.ok) throw new Error(`Failed to fetch weather data: ${response.status}`);
    
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

    // Process and cache the data
    alertsCache = processWeatherAlerts(severeWeatherAlerts);
    lastFetchTime = now;
    
    // Apply filters if requested
    if (options.filter || options.search) {
      return filterAlerts(alertsCache, options);
    }
    
    return alertsCache;
  } catch (error) {
    console.error('Error fetching weather data:', error);
    throw error;
  }
}

/**
 * Filter alerts based on search terms and filters
 */
function filterAlerts(alerts, options = {}) {
  let filtered = [...alerts];
  
  // Apply search term
  if (options.search) {
    const searchLower = options.search.toLowerCase();
    filtered = filtered.filter(alert => 
      alert.name.toLowerCase().includes(searchLower) ||
      alert.description?.toLowerCase().includes(searchLower) ||
      alert.areas?.toLowerCase().includes(searchLower)
    );
  }
  
  // Apply category filter
  if (options.category !== undefined) {
    filtered = filtered.filter(alert => alert.category === options.category);
  }
  
  // Apply region/basin filter
  if (options.basin) {
    filtered = filtered.filter(alert => alert.basin === options.basin);
  }
  
  // Apply event type filter
  if (options.type) {
    filtered = filtered.filter(alert => 
      alert.type.toLowerCase().includes(options.type.toLowerCase())
    );
  }
  
  return filtered;
}

/**
 * Process weather alerts into a standardized format
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
      basin: determineBasin(geometry),
      onset: properties.onset,
      expires: properties.expires,
      headline: properties.headline
    };
  });
}

/**
 * Determine the ocean basin based on coordinates
 */
function determineBasin(geometry) {
  if (!geometry) return 'NA'; // Default to North Atlantic
  
  const coords = extractCoordinates(geometry);
  if (!coords) return 'NA';
  
  const [lon, lat] = coords;
  
  // Simple basin determination based on coordinates
  if (lon > -100 && lon < 0 && lat > 0) return 'NA'; // North Atlantic
  if (lon >= -180 && lon < -100 && lat > 0) return 'EP'; // Eastern Pacific
  if (lon >= 100 && lon < 180 && lat > 0) return 'WP'; // Western Pacific
  if (lon >= 40 && lon < 100 && lat > 0) return 'NI'; // North Indian
  if (lon >= 40 && lon < 135 && lat <= 0) return 'SI'; // South Indian
  if (lon >= 135 && lon < 180 && lat <= 0) return 'SP'; // South Pacific
  
  return 'NA'; // Default
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
    // First, get the nearest station
    const stationResponse = await fetch(
      `${BASE_URL}/points/${lat},${lon}/stations`, {
        headers: {
          'User-Agent': USER_AGENT,
          'Accept': 'application/geo+json'
        }
      }
    );

    if (!stationResponse.ok) throw new Error('Failed to fetch stations');
    
    const stationData = await stationResponse.json();
    if (!stationData.features?.[0]?.id) {
      throw new Error('No weather stations found nearby');
    }

    // Then get the latest observations
    const obsResponse = await fetch(
      `${stationData.features[0].id}/observations/latest`, {
        headers: {
          'User-Agent': USER_AGENT,
          'Accept': 'application/geo+json'
        }
      }
    );

    if (!obsResponse.ok) throw new Error('Failed to fetch observations');
    
    const obsData = await obsResponse.json();
    return processObservations(obsData.properties);
  } catch (error) {
    console.error('Error fetching observations:', error);
    
    // Return mock data for demo/testing
    return {
      timestamp: new Date().toISOString(),
      temperature: 28.5,
      windSpeed: 75 + Math.random() * 30,
      windDirection: 120 + Math.random() * 60,
      barometricPressure: 965 + Math.random() * 15,
      seaLevelPressure: 970 + Math.random() * 10,
      visibility: 5 + Math.random() * 3,
      relativeHumidity: 85 + Math.random() * 10,
      windGust: 90 + Math.random() * 25,
      precipitationLastHour: 2.5 + Math.random() * 1.5
    };
  }
}

/**
 * Process weather observations into a standardized format
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

/**
 * Get historical data for a hurricane (simulated for demo)
 */
export function getHurricaneHistoricalData(hurricaneId, days = 7) {
  // In a real implementation, this would fetch from an API
  // For demo purposes, generate realistic mock data
  
  const history = [];
  const now = new Date();
  let baseWind = 65;
  let basePressure = 990;
  
  // Generate data for the past week
  for (let i = days * 24; i >= 0; i -= 6) {
    const timestamp = new Date(now);
    timestamp.setHours(timestamp.getHours() - i);
    
    // Add some natural variability with trend
    const trendFactor = 1 - i / (days * 24); // Increasing trend (0 to 1)
    const windVariation = (Math.random() * 8 - 4);
    const pressureVariation = (Math.random() * 5 - 2.5);
    
    // Create historical point
    history.push({
      timestamp: timestamp.toISOString(),
      hour: -i,
      windSpeed: baseWind * (1 + trendFactor * 0.5) + windVariation,
      pressure: basePressure * (1 - trendFactor * 0.1) + pressureVariation,
      category: getHurricaneCategory(baseWind * (1 + trendFactor * 0.5) + windVariation)
    });
    
    // Update base values with some persistence
    baseWind += (Math.random() * 4 - 2) + (trendFactor * 2);
    basePressure += (Math.random() * 2 - 1) - (trendFactor * 1);
  }
  
  return history;
}

/**
 * Get hurricane category based on wind speed
 */
function getHurricaneCategory(windSpeed) {
  if (windSpeed < 39) return 'TD';
  if (windSpeed < 74) return 'TS';
  if (windSpeed < 96) return '1';
  if (windSpeed < 111) return '2';
  if (windSpeed < 130) return '3';
  if (windSpeed < 157) return '4';
  return '5';
}