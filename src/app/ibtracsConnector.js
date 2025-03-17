// ibtracsConnector.js
/**
 * Connector for IBTrACS (International Best Track Archive for Climate Stewardship) data
 * This module handles downloading, parsing, and processing global tropical cyclone data
 */

import Papa from 'papaparse';

// Configuration
const IBTRACS_BASE_URL = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv';
const IBTRACS_SINCE1980_URL = `${IBTRACS_BASE_URL}/ibtracs.since1980.list.v04r00.csv`;
const IBTRACS_LAST3YEARS_URL = `${IBTRACS_BASE_URL}/ibtracs.last3years.list.v04r00.csv`;

/**
 * Fetches IBTrACS data
 * @param {string} dataset - Dataset type: 'since1980' or 'last3years'
 * @returns {Promise<ArrayBuffer>} - Raw CSV data
 */
async function fetchIBTraCSData(dataset = 'since1980') {
  try {
    const url = dataset === 'last3years' ? IBTRACS_LAST3YEARS_URL : IBTRACS_SINCE1980_URL;
    console.log(`Fetching IBTrACS data from: ${url}`);
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch IBTrACS data: ${response.status} ${response.statusText}`);
    }
    
    return await response.arrayBuffer();
  } catch (error) {
    console.error('Error fetching IBTrACS data:', error);
    throw error;
  }
}

/**
 * Parses IBTrACS CSV data
 * @param {ArrayBuffer} data - Raw CSV data
 * @returns {Promise<Object[]>} - Parsed cyclone records
 */
async function parseIBTraCSData(data) {
  return new Promise((resolve, reject) => {
    try {
      // Convert ArrayBuffer to string
      const decoder = new TextDecoder('utf-8');
      const csvString = decoder.decode(data);
      
      Papa.parse(csvString, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          console.log(`Parsed ${results.data.length} IBTrACS records`);
          resolve(results.data);
        },
        error: (error) => {
          console.error('Error parsing CSV:', error);
          reject(error);
        }
      });
    } catch (error) {
      console.error('Error in parseIBTraCSData:', error);
      reject(error);
    }
  });
}

/**
 * Processes raw IBTrACS records into standardized hurricane data
 * @param {Object[]} records - Raw IBTrACS records
 * @returns {Object[]} - Processed hurricane data
 */
function processIBTraCSRecords(records) {
  // Group records by storm ID (each storm has multiple time points)
  const stormGroups = {};
  
  records.forEach(record => {
    // Skip records with missing key data
    if (!record.SID || 
        record.LAT === null || record.LON === null || 
        record.USA_WIND === null || record.USA_PRES === null) {
      return;
    }
    
    // Use SID (Storm ID) as the grouping key
    const stormId = record.SID;
    
    if (!stormGroups[stormId]) {
      stormGroups[stormId] = [];
    }
    
    // Convert ISO time format
    let timestamp;
    try {
      if (record.ISO_TIME) {
        timestamp = new Date(record.ISO_TIME);
      } else {
        // Fallback if ISO_TIME is missing - construct from year/month/day/hour
        timestamp = new Date(
          record.SEASON || 2000, 
          (record.MONTH || 1) - 1, 
          record.DAY || 1, 
          record.HOUR || 0
        );
      }
    } catch (e) {
      console.warn(`Invalid date for storm ${stormId}:`, e);
      timestamp = new Date(); // Fallback to prevent crashes
    }
    
    // Process single time point
    stormGroups[stormId].push({
      // Position data
      position: { 
        lat: record.LAT, 
        lon: record.LON 
      },
      // Wind speeds are in knots, convert to mph
      windSpeed: record.USA_WIND !== null ? record.USA_WIND * 1.15078 : null,
      // Pressure in millibars/hPa
      pressure: record.USA_PRES,
      // Storm status (may need normalization)
      status: record.USA_STATUS || record.NATURE,
      // Record time
      timestamp: timestamp,
      // Radius of maximum winds (if available)
      rmw: record.USA_RMW,
      // Storm size - radius of 34kt winds (if available)
      r34: record.USA_R34,
      // Basin
      basin: record.BASIN
    });
  });
  
  // Convert storm groups to processed hurricanes
  const processedHurricanes = Object.entries(stormGroups).map(([id, track]) => {
    // Sort track by timestamp
    const sortedTrack = [...track].sort((a, b) => 
      a.timestamp.getTime() - b.timestamp.getTime()
    );
    
    // Get name from record
    // Find the first record with a name field
    const recordWithName = records.find(r => r.SID === id && r.NAME);
    const name = recordWithName?.NAME || `Unnamed Storm ${id}`;
    
    // Get max wind speed to determine category
    const maxWindSpeed = Math.max(...sortedTrack.map(p => p.windSpeed || 0));
    const category = getHurricaneCategoryFromWindSpeed(maxWindSpeed);
    
    // Get year/season
    const year = sortedTrack[0]?.timestamp.getFullYear() || new Date().getFullYear();
    
    return {
      id,
      name,
      year,
      category,
      basin: sortedTrack[0]?.basin,
      startTime: sortedTrack[0]?.timestamp,
      initialPosition: sortedTrack[0]?.position,
      initialWindSpeed: sortedTrack[0]?.windSpeed,
      initialPressure: sortedTrack[0]?.pressure,
      track: sortedTrack
    };
  });
  
  // Ensure all hurricane IDs are unique by adding a suffix if needed
  const uniqueIdMap = new Map();
  const uniqueHurricanes = [];
  
  processedHurricanes.forEach(hurricane => {
    let uniqueId = hurricane.id;
    
    // Check if we've seen this ID before
    if (uniqueIdMap.has(uniqueId)) {
      const count = uniqueIdMap.get(uniqueId) + 1;
      uniqueIdMap.set(uniqueId, count);
      uniqueId = `${hurricane.id}-${count}`;
    } else {
      uniqueIdMap.set(uniqueId, 1);
    }
    
    // Create a copy with the unique ID
    uniqueHurricanes.push({
      ...hurricane,
      id: uniqueId
    });
  });
  
  return uniqueHurricanes;
}

/**
 * Gets hurricane category from wind speed in mph
 * @param {number} windSpeed - Wind speed in mph
 * @returns {number} - Hurricane category (0-5, where 0 is tropical storm)
 */
function getHurricaneCategoryFromWindSpeed(windSpeed) {
  if (windSpeed < 74) return 0; // Tropical Depression or Tropical Storm
  if (windSpeed < 96) return 1;
  if (windSpeed < 111) return 2;
  if (windSpeed < 130) return 3;
  if (windSpeed < 157) return 4;
  return 5;
}

/**
 * Filters hurricanes based on criteria
 * @param {Object[]} hurricanes - Processed hurricane data
 * @param {Object} options - Filter options
 * @returns {Object[]} - Filtered hurricane data
 */
function filterHurricanes(hurricanes, options = {}) {
  let filtered = [...hurricanes];
  
  // Filter by basin
  if (options.basin) {
    filtered = filtered.filter(h => h.basin === options.basin);
  }
  
  // Filter by year/season
  if (options.year) {
    filtered = filtered.filter(h => h.year === options.year);
  }
  
  // Filter by minimum category
  if (options.minCategory !== undefined) {
    filtered = filtered.filter(h => h.category >= options.minCategory);
  }
  
  // Filter to ensure adequate track data
  if (options.minTrackPoints) {
    filtered = filtered.filter(h => h.track.length >= options.minTrackPoints);
  }
  
  return filtered;
}

/**
 * Main function to fetch and process IBTrACS data
 * @param {Object} options - Options for data fetching and filtering
 * @returns {Promise<Object[]>} - Processed hurricane data
 */
export async function fetchIBTraCSHurricanes(options = {}) {
  try {
    // Default to recent data for faster loading during development
    const dataset = options.fullHistory ? 'since1980' : 'last3years';
    
    // Fetch raw data
    const rawData = await fetchIBTraCSData(dataset);
    
    // Parse CSV data
    const records = await parseIBTraCSData(rawData);
    
    // Process records into hurricane objects
    const hurricanes = processIBTraCSRecords(records);
    
    // Apply filters
    return filterHurricanes(hurricanes, options);
  } catch (error) {
    console.error('Error in fetchIBTraCSHurricanes:', error);
    throw error;
  }
}

/**
 * Utility function to get basins from processed hurricanes
 * @param {Object[]} hurricanes - Processed hurricane data
 * @returns {Object} - Object with basin codes and counts
 */
export function getBasinStats(hurricanes) {
  const basinCounts = {};
  const basinNames = {
    NA: 'North Atlantic',
    SA: 'South Atlantic',
    EP: 'Eastern Pacific',
    WP: 'Western Pacific',
    NI: 'North Indian',
    SI: 'South Indian',
    SP: 'South Pacific',
    MM: 'Multi-basin' // For storms that cross between basins
  };
  
  hurricanes.forEach(h => {
    const basin = h.basin || 'UNKNOWN';
    basinCounts[basin] = (basinCounts[basin] || 0) + 1;
  });
  
  return {
    counts: basinCounts,
    names: basinNames
  };
}