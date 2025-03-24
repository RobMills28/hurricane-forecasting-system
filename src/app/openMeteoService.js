// Enhanced openMeteoService.js for global storm tracking

/**
 * Service to handle Open-Meteo API interactions
 * Provides access to multiple weather agencies' data through a unified interface
 * Enhanced to detect severe storms globally
 */

const BASE_URL = 'https://api.open-meteo.com/v1';

// Mapping of regions to their specific Open-Meteo endpoints
const REGION_CONFIG = {
  GLOBAL: {
    endpoint: `${BASE_URL}/forecast`,
    params: {
      hourly: 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation,weather_code',
      daily: 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant'
    }
  }
};

// Expanded Weather Codes for severe weather detection
const SEVERE_WEATHER_CODES = {
  // Thunderstorms
  95: "Thunderstorm with slight or moderate hail",
  96: "Thunderstorm with slight or moderate hail",
  97: "Heavy thunderstorm",
  98: "Thunderstorm with duststorm",
  99: "Thunderstorm with heavy hail",
  
  // Heavy Rain
  91: "Rain shower(s), heavy",
  92: "Heavy rain at time of observation",
  
  // Snow/Ice storms
  85: "Snow shower(s), moderate or heavy",
  86: "Snow shower(s), moderate or heavy",
  
  // Only include codes that represent significant weather events
};

// Intensity thresholds for various storm types
const STORM_THRESHOLDS = {
  // Tropical system thresholds (wind speeds in km/h)
  TROPICAL_DEPRESSION: { windSpeed: 35, pressure: 1005 }, // 35-63 km/h 
  TROPICAL_STORM: { windSpeed: 63, pressure: 1000 },      // 63-118 km/h
  HURRICANE_1: { windSpeed: 118, pressure: 980 },         // 118-154 km/h
  HURRICANE_2: { windSpeed: 154, pressure: 965 },         // 154-177 km/h
  HURRICANE_3: { windSpeed: 177, pressure: 945 },         // 177-209 km/h
  HURRICANE_4: { windSpeed: 209, pressure: 920 },         // 209-252 km/h
  HURRICANE_5: { windSpeed: 252, pressure: 920 },         // >252 km/h
  
  // Severe non-tropical storms
  SEVERE_THUNDERSTORM: { windSpeed: 90, pressure: null }, // >90 km/h winds
  WINTER_STORM: { windSpeed: 50, pressure: 990 },         // Strong winter storm
  
  // Regional naming
  REGIONS: {
    'WP': 'Typhoon',    // Western Pacific
    'NI': 'Cyclone',    // North Indian
    'SI': 'Cyclone',    // South Indian
    'SP': 'Cyclone',    // South Pacific
    'NA': 'Hurricane',  // North Atlantic
    'EP': 'Hurricane'   // Eastern Pacific
  }
};

// Global Storm Hotspots - comprehensive coverage of all ocean basins
const GLOBAL_STORM_HOTSPOTS = {
  // Western Pacific (Typhoons)
  'WP': [
    { latitude: 15.0, longitude: 130.0, name: 'Philippine Sea' },
    { latitude: 20.0, longitude: 135.0, name: 'Western Pacific' },
    { latitude: 25.0, longitude: 125.0, name: 'East China Sea' },
    { latitude: 10.0, longitude: 145.0, name: 'Micronesia Region' },
    { latitude: 12.0, longitude: 125.0, name: 'Philippines Region' }
  ],
  
  // North Atlantic (Hurricanes)
  'NA': [
    { latitude: 25.0, longitude: -75.0, name: 'Western Atlantic' },
    { latitude: 15.0, longitude: -55.0, name: 'Central Atlantic' },
    { latitude: 12.0, longitude: -40.0, name: 'Eastern Atlantic' },
    { latitude: 20.0, longitude: -85.0, name: 'Caribbean Sea' },
    { latitude: 25.0, longitude: -90.0, name: 'Gulf of Mexico' }
  ],
  
  // Eastern Pacific (Hurricanes)
  'EP': [
    { latitude: 15.0, longitude: -105.0, name: 'Eastern Pacific' },
    { latitude: 12.0, longitude: -120.0, name: 'Central Pacific' }
  ],
  
  // North Indian Ocean (Cyclones)
  'NI': [
    { latitude: 15.0, longitude: 85.0, name: 'Bay of Bengal' },
    { latitude: 15.0, longitude: 65.0, name: 'Arabian Sea' }
  ],
  
  // South Indian Ocean (Cyclones)
  'SI': [
    { latitude: -15.0, longitude: 70.0, name: 'Madagascar Basin' },
    { latitude: -15.0, longitude: 90.0, name: 'Eastern Indian Ocean' }
  ],
  
  // South Pacific (Cyclones)
  'SP': [
    { latitude: -15.0, longitude: 170.0, name: 'Fiji Basin' },
    { latitude: -15.0, longitude: 145.0, name: 'Coral Sea' }
  ],
  
  // Severe non-tropical storm regions
  'EU': [
    { latitude: 55.0, longitude: 0.0, name: 'North Atlantic' },
    { latitude: 45.0, longitude: 10.0, name: 'Mediterranean' },
    { latitude: 60.0, longitude: 15.0, name: 'Scandinavia' }
  ],
  
  'NA_WINTER': [
    { latitude: 45.0, longitude: -90.0, name: 'US Midwest' },
    { latitude: 40.0, longitude: -75.0, name: 'US Northeast' },
    { latitude: 50.0, longitude: -100.0, name: 'Canadian Prairies' }
  ],
  
  'ASIA_WINTER': [
    { latitude: 45.0, longitude: 125.0, name: 'Northeast Asia' },
    { latitude: 35.0, longitude: 140.0, name: 'Japan' }
  ],
  
  'EUROPE_WINTER': [
    { latitude: 55.0, longitude: 10.0, name: 'Northern Europe' },
    { latitude: 47.0, longitude: 10.0, name: 'Alpine Region' },
    { latitude: 53.0, longitude: -2.0, name: 'British Isles' }
  ]
};

// Cache for responses to minimize API calls
const responseCache = new Map();
const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes

/**
 * Get forecast data for a specific location using the appropriate regional model
 * @param {number} latitude - Latitude of the location
 * @param {number} longitude - Longitude of the location
 * @param {string} region - Region identifier (WP, NA, etc.)
 * @param {Object} options - Additional options
 * @returns {Promise<Object>} Forecast data
 */
export async function getRegionalForecast(latitude, longitude, region = 'GLOBAL', options = {}) {
  try {
    // Use the global configuration
    const config = REGION_CONFIG.GLOBAL;
    
    // Generate cache key based on parameters
    const cacheKey = `${region}_${latitude}_${longitude}_${options.past_days || 0}_${options.forecast_days || 7}`;
    
    // Check cache first
    const now = Date.now();
    if (responseCache.has(cacheKey)) {
      const cachedData = responseCache.get(cacheKey);
      if (now - cachedData.timestamp < CACHE_DURATION) {
        console.log(`Using cached forecast data for ${region}`);
        return cachedData.data;
      }
    }
    
    // Construct query parameters
    const params = new URLSearchParams({
      latitude: latitude.toString(),
      longitude: longitude.toString(),
      ...config.params,
      timezone: options.timezone || 'auto',
      forecast_days: options.forecast_days?.toString() || '7',
      past_days: options.past_days?.toString() || '0'
    });
    
    // Make API request
    const response = await fetch(`${config.endpoint}?${params.toString()}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to fetch forecast: ${response.status} - ${errorText}`);
    }
    
    const data = await response.json();
    
    // Cache the result
    responseCache.set(cacheKey, {
      data,
      timestamp: now
    });
    
    // Process the data to match your internal structures
    return processOpenMeteoData(data, region);
  } catch (error) {
    console.error(`Error fetching ${region} forecast:`, error);
    throw error;
  }
}

/**
 * Process Open-Meteo data into a standardized format for your application
 * @param {Object} data - Raw Open-Meteo data
 * @param {string} region - Region identifier
 * @returns {Object} Processed data in your app's format
 */
function processOpenMeteoData(data, region) {
  try {
    // Extract relevant data
    const { hourly, daily, hourly_units, daily_units } = data;
    
    // Skip processing if essential data is missing
    if (!hourly || !hourly.time || !hourly.weather_code) {
      throw new Error('Essential forecast data missing from response');
    }
    
    // Convert to your application's format
    const processed = {
      source: `Open-Meteo-${region}`,
      coordinates: [data.longitude, data.latitude],
      elevation: data.elevation,
      timezone: data.timezone,
      
      // Find any storms in the forecast data
      storms: detectStorms(hourly, daily, region),
      
      // Raw hourly forecast data with appropriate units
      hourlyForecast: formatHourlyData(hourly, hourly_units),
      
      // Raw daily forecast data with appropriate units
      dailyForecast: daily ? formatDailyData(daily, daily_units) : null,
      
      // Original data for reference if needed
      rawData: data
    };
    
    return processed;
  } catch (error) {
    console.error('Error processing Open-Meteo data:', error);
    throw error;
  }
}

/**
 * Format hourly data into a more usable structure
 * @param {Object} hourly - Raw hourly data
 * @param {Object} units - Units for each variable
 * @returns {Array} Formatted hourly data
 */
function formatHourlyData(hourly, units) {
  const timePoints = hourly.time.length;
  const formattedData = [];
  
  for (let i = 0; i < timePoints; i++) {
    const point = {
      time: hourly.time[i],
      
      // Essential weather variables
      temperature: hourly.temperature_2m?.[i],
      humidity: hourly.relative_humidity_2m?.[i],
      pressure: hourly.pressure_msl?.[i],
      windSpeed: hourly.wind_speed_10m?.[i],
      windDirection: hourly.wind_direction_10m?.[i],
      precipitation: hourly.precipitation?.[i],
      weatherCode: hourly.weather_code?.[i],
      
      // Units
      units: {
        temperature: units.temperature_2m,
        humidity: units.relative_humidity_2m,
        pressure: units.pressure_msl,
        windSpeed: units.wind_speed_10m,
        windDirection: units.wind_direction_10m,
        precipitation: units.precipitation,
      }
    };
    
    formattedData.push(point);
  }
  
  return formattedData;
}

/**
 * Format daily data into a more usable structure
 * @param {Object} daily - Raw daily data
 * @param {Object} units - Units for each variable
 * @returns {Array} Formatted daily data
 */
function formatDailyData(daily, units) {
  const days = daily.time.length;
  const formattedData = [];
  
  for (let i = 0; i < days; i++) {
    const point = {
      date: daily.time[i],
      
      // Essential daily weather variables
      weatherCode: daily.weather_code?.[i],
      maxTemperature: daily.temperature_2m_max?.[i],
      minTemperature: daily.temperature_2m_min?.[i],
      precipitationSum: daily.precipitation_sum?.[i],
      maxWindSpeed: daily.wind_speed_10m_max?.[i],
      dominantWindDirection: daily.wind_direction_10m_dominant?.[i],
      
      // Units
      units: {
        maxTemperature: units.temperature_2m_max,
        minTemperature: units.temperature_2m_min,
        precipitationSum: units.precipitation_sum,
        maxWindSpeed: units.wind_speed_10m_max,
        dominantWindDirection: units.wind_direction_10m_dominant
      }
    };
    
    formattedData.push(point);
  }
  
  return formattedData;
}

/**
 * Enhanced storm detection function that identifies all types of severe storms
 * @param {Object} hourly - Hourly forecast data
 * @param {Object} daily - Daily forecast data
 * @param {string} region - Ocean basin or region identifier
 * @returns {Array} Detected storms with classification
 */
function detectStorms(hourly, daily, region) {
  const storms = [];
  
  // Check for continuous periods of severe weather with strong winds
  for (let i = 0; i < hourly.time.length; i++) {
    // Skip if we don't have all the data we need
    if (!hourly.weather_code?.[i] || 
        !hourly.wind_speed_10m?.[i] || 
        !hourly.pressure_msl?.[i]) {
      continue;
    }
    
    const weatherCode = hourly.weather_code[i];
    const windSpeed = hourly.wind_speed_10m[i];
    const pressure = hourly.pressure_msl[i];
    const time = hourly.time[i];
    
    // Determine if this is a severe weather event based on weather code
    const isSevereWeatherCode = Object.keys(SEVERE_WEATHER_CODES).includes(weatherCode.toString());
    
    // Check for tropical cyclone criteria (strong winds + low pressure)
    const isTropicalCycloneWind = windSpeed >= STORM_THRESHOLDS.TROPICAL_DEPRESSION.windSpeed;
    const isLowPressure = pressure < STORM_THRESHOLDS.TROPICAL_STORM.pressure;
    
    // Check for severe winter storm (strong winds + cold temperatures)
    const temperature = hourly.temperature_2m?.[i];
    const isWinterStorm = (
      windSpeed >= STORM_THRESHOLDS.WINTER_STORM.windSpeed && 
      temperature !== undefined && 
      temperature < 2 // Below 2°C 
    );
    
    // Check for severe thunderstorm (strong winds + any thunderstorm code)
    const isThunderstorm = weatherCode >= 95 && weatherCode <= 99;
    const isSevereThunderstorm = windSpeed >= STORM_THRESHOLDS.SEVERE_THUNDERSTORM.windSpeed && isThunderstorm;
    
    // If any severe weather condition is met
    if (isTropicalCycloneWind || isSevereThunderstorm || isWinterStorm || 
        (isSevereWeatherCode && windSpeed >= 50)) {
      
      // Check if this is a continuation of an existing storm
      const existingStorm = storms.find(s => {
        const lastTimeIndex = hourly.time.indexOf(s.lastObservedTime);
        return Math.abs(i - lastTimeIndex) <= 6; // Within 6 hours
      });
      
      if (existingStorm) {
        // Update existing storm
        existingStorm.lastObservedTime = time;
        existingStorm.observations.push({
          time,
          windSpeed,
          pressure,
          weatherCode,
          temperature
        });
        
        // Update max values if needed
        if (windSpeed > existingStorm.maxWindSpeed) {
          existingStorm.maxWindSpeed = windSpeed;
        }
        if (pressure < existingStorm.minPressure) {
          existingStorm.minPressure = pressure;
        }
      } else {
        // Create new storm detection
        storms.push({
          id: `storm-${time.replace(/[:.]/g, '-')}`,
          firstObservedTime: time,
          lastObservedTime: time,
          maxWindSpeed: windSpeed,
          minPressure: pressure,
          observations: [{
            time,
            windSpeed,
            pressure,
            weatherCode,
            temperature
          }],
          region
        });
      }
    }
  }
  
  // Process the detected storms and classify them
  return storms.map(storm => {
    // Classify storm based on characteristics
    const classification = classifyStorm(storm.maxWindSpeed, storm.minPressure, storm.region);
    
    // Get a more appropriate name based on the classification
    const name = generateStormName(classification.stormType, storm.id, storm.region);
    
    return {
      id: storm.id,
      type: classification.stormType,
      category: classification.category,
      name,
      firstObservedTime: storm.firstObservedTime,
      lastObservedTime: storm.lastObservedTime,
      maxWindSpeed: storm.maxWindSpeed,
      minPressure: storm.minPressure,
      observations: storm.observations,
      region: storm.region,
      basin: determineBasin(storm.region)
    };
  });
}

/**
 * Classify a storm based on its characteristics
 * @param {number} windSpeed - Maximum wind speed in km/h
 * @param {number} pressure - Minimum pressure in hPa
 * @param {string} region - Region/basin identifier
 * @returns {Object} Classification with storm type and category
 */
function classifyStorm(windSpeed, pressure, region) {
  // Default classification
  let stormType = 'Severe Storm';
  let category = null;
  
  // Determine basin name from region if possible
  const basin = determineBasin(region);
  
  // Determine region-specific naming
  const regionPrefix = STORM_THRESHOLDS.REGIONS[basin] || 'Severe Storm';
  
  // Winter storm special handling
  if ((region && (region.includes('WINTER') || region === 'EU' || region === 'EUROPE_WINTER') || 
       basin === 'NA' && new Date().getMonth() >= 10 || new Date().getMonth() <= 2) && 
      windSpeed >= STORM_THRESHOLDS.WINTER_STORM.windSpeed) {
    
    stormType = 'Winter Storm';
    
    // Categorize winter storms based on wind speed
    if (windSpeed >= 90) category = '4';
    else if (windSpeed >= 70) category = '3';
    else if (windSpeed >= 60) category = '2';
    else category = '1';
  } 
  // Tropical cyclone classification
  else if (windSpeed >= STORM_THRESHOLDS.HURRICANE_1.windSpeed) {
    // Classify based on hurricanes/typhoons/cyclones categories
    if (windSpeed >= STORM_THRESHOLDS.HURRICANE_5.windSpeed) {
      category = '5';
    } else if (windSpeed >= STORM_THRESHOLDS.HURRICANE_4.windSpeed) {
      category = '4';
    } else if (windSpeed >= STORM_THRESHOLDS.HURRICANE_3.windSpeed) {
      category = '3';
    } else if (windSpeed >= STORM_THRESHOLDS.HURRICANE_2.windSpeed) {
      category = '2';
    } else {
      category = '1';
    }
    
    stormType = regionPrefix;
  } 
  // Tropical storm
  else if (windSpeed >= STORM_THRESHOLDS.TROPICAL_STORM.windSpeed) {
    stormType = 'Tropical Storm';
    category = 'TS';
  } 
  // Tropical depression
  else if (windSpeed >= STORM_THRESHOLDS.TROPICAL_DEPRESSION.windSpeed) {
    stormType = 'Tropical Depression';
    category = 'TD';
  }
  // Severe thunderstorm
  else if (windSpeed >= STORM_THRESHOLDS.SEVERE_THUNDERSTORM.windSpeed) {
    stormType = 'Severe Thunderstorm';
    category = windSpeed >= 100 ? '3' : windSpeed >= 90 ? '2' : '1';
  }
  // Default severe storm
  else {
    stormType = 'Severe Storm';
    category = '1';
  }
  
  return { stormType, category };
}

/**
 * Determine the ocean basin for a given region code or coordinates
 * @param {string|Array} regionOrCoords - Region code or [lon, lat] coordinates
 * @returns {string} Basin code (NA, WP, etc.)
 */
function determineBasin(regionOrCoords) {
  // If it's already a basin code, return it
  if (typeof regionOrCoords === 'string' && 
      ['NA', 'EP', 'WP', 'NI', 'SI', 'SP'].includes(regionOrCoords)) {
    return regionOrCoords;
  }
  
  // If it's a region that maps to a basin
  if (typeof regionOrCoords === 'string') {
    if (regionOrCoords.includes('JAPAN') || regionOrCoords.includes('ASIA')) return 'WP';
    if (regionOrCoords.includes('AUSTRALIA')) return 'SP';
    if (regionOrCoords.includes('INDIA')) return 'NI';
    if (regionOrCoords.includes('EUROPE')) return 'NA';
    if (regionOrCoords.includes('AFRICA')) return 'SI';
    if (regionOrCoords.includes('AMERICA')) return 'NA';
  }
  
  // If coordinates are provided, determine basin by location
  if (Array.isArray(regionOrCoords) && regionOrCoords.length === 2) {
    const [lon, lat] = regionOrCoords;
    
    // Simple basin determination based on coordinates
    if (lon > -100 && lon < 0 && lat > 0) return 'NA'; // North Atlantic
    if (lon >= -180 && lon < -100 && lat > 0) return 'EP'; // Eastern Pacific
    if (lon >= 100 && lon < 180 && lat > 0) return 'WP'; // Western Pacific
    if (lon >= 40 && lon < 100 && lat > 0) return 'NI'; // North Indian
    if (lon >= 40 && lon < 135 && lat <= 0) return 'SI'; // South Indian
    if (lon >= 135 && lon < 180 && lat <= 0) return 'SP'; // South Pacific
  }
  
  return 'NA'; // Default to North Atlantic if unknown
}

/**
 * Generate a name for a storm based on type and region
 * @param {string} stormType - Type of storm (Hurricane, Typhoon, etc.)
 * @param {string} stormId - Unique identifier for the storm
 * @param {string} region - Region/basin code
 * @returns {string} Storm name
 */
function generateStormName(stormType, stormId, region) {
  // For now, just use a generic name based on the storm type and ID
  // In a real system, this would use official naming conventions or lists
  
  const adjectives = [
    'Intense', 'Powerful', 'Massive', 'Severe', 'Strong',
    'Extreme', 'Major', 'Dangerous', 'Wild', 'Fierce'
  ];
  
  // Use a deterministic approach to select an adjective based on storm ID
  // so the same storm always gets the same name
  const hash = stormId.split('').reduce((acc, char) => {
    return acc + char.charCodeAt(0);
  }, 0);
  
  const adjectiveIndex = hash % adjectives.length;
  const adjective = adjectives[adjectiveIndex];
  
  // Generate a simple ID suffix
  const idSuffix = stormId.slice(-5).replace(/[^a-zA-Z0-9]/g, '');
  
  // Create the storm name
  return `${adjective} ${stormType} ${idSuffix}`;
}

/**
 * Fetch observations for a specific location
 * @param {number} latitude - Latitude
 * @param {number} longitude - Longitude
 * @param {string} region - Region code
 * @returns {Promise<Object>} Weather observations
 */
export async function getOpenMeteoObservations(latitude, longitude, region = 'GLOBAL') {
  try {
    // Use the regional forecast to get current data
    const forecast = await getRegionalForecast(latitude, longitude, region, { past_days: 1 });
    
    // Extract the most recent data point as observations
    const hourlyData = forecast.hourlyForecast || [];
    const currentHour = new Date().getHours();
    
    // Find the closest hour data
    const closestHourData = hourlyData.reduce((closest, point) => {
      if (!closest) return point;
      
      const pointHour = new Date(point.time).getHours();
      const closestHour = new Date(closest.time).getHours();
      
      const pointDiff = Math.abs(pointHour - currentHour);
      const closestDiff = Math.abs(closestHour - currentHour);
      
      return pointDiff < closestDiff ? point : closest;
    }, null);
    
    if (!closestHourData) {
      throw new Error('No current observations available');
    }
    
    // Format observations to match expected structure
    return {
      timestamp: closestHourData.time,
      temperature: closestHourData.temperature,
      windSpeed: closestHourData.windSpeed,
      windDirection: closestHourData.windDirection,
      barometricPressure: closestHourData.pressure,
      relativeHumidity: closestHourData.humidity,
      precipitationLastHour: closestHourData.precipitation,
      weatherCode: closestHourData.weatherCode,
      
      // Match the units from the forecast
      units: closestHourData.units
    };
  } catch (error) {
    console.error('Error fetching observations:', error);
    throw error;
  }
}

/**
 * Format a detected storm in the standard hurricane format
 * @param {Object} storm - Storm data from Open-Meteo
 * @param {Object} hotspot - The hotspot where this storm was detected
 * @param {string} regionCode - Region code identifier
 * @returns {Object} Storm formatted to match hurricane format
 */
function formatStormAsHurricane(storm, hotspot, regionCode) {
  // Get the basin based on region and determine appropriate naming
  const basin = determineBasin(regionCode);
  
  // Convert km/h to mph for consistent units with NOAA data
  const windSpeedMph = storm.maxWindSpeed * 0.621371;
  
  // Create a better description of the affected area
  const areasDescription = `${hotspot.name} region`;
  
  // Map from Open-Meteo format to your application's hurricane format
  return {
    id: storm.id,
    name: storm.name,
    type: storm.type,
    severity: getSeverityFromCategory(storm.category),
    certainty: getCertaintyFromObservations(storm.observations),
    basin: basin,
    category: storm.category,
    coordinates: [hotspot.longitude, hotspot.latitude], // [lon, lat]
    windSpeed: windSpeedMph,
    pressure: storm.minPressure,
    areas: areasDescription,
    status: 'Active',
    onset: storm.firstObservedTime,
    expires: getExpirationTime(storm.lastObservedTime),
    dataSource: regionCode
    // Removed visualRadius property to fix the display issue
  };
}

/**
 * Determine severity level from storm category
 * @param {string|number} category - Storm category
 * @returns {string} Severity level
 */
function getSeverityFromCategory(category) {
  if (category === 'TD') return 'Minor';
  if (category === 'TS') return 'Moderate';
  
  const categoryNum = parseInt(category);
  if (categoryNum >= 4) return 'Extreme';
  if (categoryNum >= 3) return 'Severe';
  if (categoryNum >= 1) return 'Moderate';
  
  return 'Minor';
}

/**
 * Determine certainty based on number and consistency of observations
 * @param {Array} observations - Storm observations
 * @returns {string} Certainty level
 */
function getCertaintyFromObservations(observations) {
  if (!observations || observations.length === 0) return 'Possible';
  
  // More observations = more certain
  if (observations.length >= 5) return 'Observed';
  if (observations.length >= 3) return 'Likely';
  
  return 'Possible';
}

/**
 * Calculate expiration time for storm alerts
 * @param {string} lastTime - ISO timestamp of last observation
 * @returns {string} ISO timestamp of expiration
 */
function getExpirationTime(lastTime) {
  // Default forecast valid for 24 hours after last observation
  const expireDate = new Date(lastTime);
  expireDate.setHours(expireDate.getHours() + 24);
  return expireDate.toISOString();
}

/**
 * Get active severe weather events globally by region
 * Polls hotspots for each region to find severe storms
 * @param {string} region - Optional region code to filter results
 * @returns {Promise<Array>} Active storms in standardized format
 */
export async function getActiveHurricanesByRegion(region = 'GLOBAL') {
  try {
    const allStorms = [];
    
    // Determine which regions to check
    const regionsToCheck = region === 'GLOBAL' 
      ? Object.keys(GLOBAL_STORM_HOTSPOTS)
      : [region];
    
    // Filter to only check requested regions or those that match a partial name
    const filteredRegions = Object.keys(GLOBAL_STORM_HOTSPOTS).filter(r => {
      if (regionsToCheck.includes(r)) return true;
      if (region !== 'GLOBAL' && r.includes(region)) return true;
      return false;
    });
    
    if (filteredRegions.length === 0) {
      // If no regions match, default to some common regions
      filteredRegions.push('WP', 'NA', 'EP', 'EU');
    }
    
    // For each region, check all hotspots
    for (const regionCode of filteredRegions) {
      const hotspots = GLOBAL_STORM_HOTSPOTS[regionCode] || [];
      
      // Check each hotspot in parallel
      const regionPromises = hotspots.map(async hotspot => {
        try {
          const forecast = await getRegionalForecast(
            hotspot.latitude, 
            hotspot.longitude, 
            regionCode
          );
          
          // Get any storms detected in this forecast
          const storms = forecast.storms || [];
          
          // Format storms to match the expected hurricane format
          return storms.map(storm => formatStormAsHurricane(storm, hotspot, regionCode));
        } catch (error) {
          console.error(`Error checking hotspot ${hotspot.name}:`, error);
          return [];
        }
      });
      
      // Wait for all hotspots in this region to be checked
      const regionResults = await Promise.all(regionPromises);
      
      // Flatten results and add to all storms
      regionResults.forEach(stormList => {
        if (stormList && stormList.length > 0) {
          allStorms.push(...stormList);
        }
      });
    }
    
    // Remove duplicates by ID
    const uniqueStorms = [];
    const stormIds = new Set();
    
    allStorms.forEach(storm => {
      if (!stormIds.has(storm.id)) {
        stormIds.add(storm.id);
        uniqueStorms.push(storm);
      }
    });
    
    return uniqueStorms;
  } catch (error) {
    console.error('Error fetching active hurricanes by region:', error);
    throw error;
  }
}