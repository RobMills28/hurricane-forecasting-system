// nasaService.js
// Service to handle NASA API interactions

const NASA_API_KEY = 'DEMO_KEY'; // Replace with your NASA API key for production use
const NASA_API_BASE_URL = 'https://api.nasa.gov/planetary/earth';

/**
 * Fetches NASA Earth imagery assets for a given location and date
 * 
 * @param {number} lat - Latitude
 * @param {number} lon - Longitude
 * @param {string} date - Date in YYYY-MM-DD format
 * @param {number} dim - Width and height of image in degrees (0.025 to 0.5)
 * @returns {Promise<Object>} The assets data with URL to the satellite image
 */
export async function getNasaImageryAssets(lat, lon, date = new Date().toISOString().split('T')[0], dim = 0.15) {
  try {
    const response = await fetch(
      `${NASA_API_BASE_URL}/assets?lat=${lat}&lon=${lon}&date=${date}&dim=${dim}&api_key=${NASA_API_KEY}`
    );
    
    if (!response.ok) {
      throw new Error(`NASA API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching NASA imagery assets:', error);
    throw error;
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
    // NASA GIBS doesn't have a direct API to list layers, so we're returning a curated list
    // In a production app, you might fetch this from a more dynamic source
    return [
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
      }
    ];
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