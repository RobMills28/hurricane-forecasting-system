'use client';

import dynamic from 'next/dynamic';
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { AlertTriangle, Wind, Droplets, Navigation, ThermometerSun, ChevronDown, ChevronUp, Info, Clock, Share, Download, Eye, ArrowUpRight, BarChart4, Globe } from 'lucide-react';
import { getActiveHurricanes, getHurricaneObservations } from './noaaService';
import { 
  getActiveHurricanesByRegion, 
  getOpenMeteoObservations, 
  getRegionalForecast 
} from './openMeteoService';
import AtlasCommandMap from './components/AtlasCommandMap';
import HurricanePrediction from './components/HurricanePrediction';
import { getNasaImagery, getGibsLayers } from './nasaService';

// For persisting data between page refreshes
const LOCAL_STORAGE_KEY = 'atlas_hurricanes_data';
const LOCAL_STORAGE_TIMESTAMP_KEY = 'atlas_hurricanes_timestamp';
const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

export default function HurricaneTracker() {
  const [hurricanes, setHurricanes] = useState([]);
  const [selectedHurricane, setSelectedHurricane] = useState(null);
  const [observations, setObservations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [forecastData, setForecastData] = useState([]);
  const [riskLevel, setRiskLevel] = useState('moderate');
  const [historyExpanded, setHistoryExpanded] = useState(false);
  const [impactExpanded, setImpactExpanded] = useState(false);
  const [detailsExpanded, setDetailsExpanded] = useState(true);
  const [forecastView, setForecastView] = useState('track'); // 'track', 'intensity', 'risk'
  const [selectedTimePeriod, setSelectedTimePeriod] = useState('24h');
  const [satelliteImagery, setSatelliteImagery] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [dataSource, setDataSource] = useState('all'); // 'us', 'japan', 'australia', 'all'
  const [isDataFetching, setIsDataFetching] = useState(false);
  const [potentialStormAreas, setPotentialStormAreas] = useState([]);


  // Load cached data on initial load
  useEffect(() => {
    try {
      const cachedData = localStorage.getItem(LOCAL_STORAGE_KEY);
      const timestamp = localStorage.getItem(LOCAL_STORAGE_TIMESTAMP_KEY);
      
      if (cachedData && timestamp) {
        const parsedTimestamp = parseInt(timestamp);
        const now = Date.now();
        
        // Use cached data if it's fresh enough
        if (now - parsedTimestamp < CACHE_DURATION) {
          const parsedData = JSON.parse(cachedData);
          setHurricanes(parsedData);
          console.log("Using cached storm data:", parsedData.length, "storms");
          setLoading(false);
          
          // Still fetch fresh data in the background after a short delay
          setTimeout(() => fetchGlobalData(true), 3000);
          return;
        }
      }
    } catch (err) {
      console.warn("Error loading cached data:", err);
    }
    
    // Fetch data if no cache or expired
    fetchGlobalData();
  }, []);

  // Set up refresh interval
  useEffect(() => {
    const interval = setInterval(() => {
      if (!isDataFetching) {
        fetchGlobalData(true);
      }
    }, 300000); // Update every 5 minutes
    
    return () => clearInterval(interval);
  }, [isDataFetching]);
  
  // Update filter effect
  useEffect(() => {
    if (!loading && !isDataFetching) {
      fetchGlobalData();
    }
  }, [dataSource]);

  // Cache data whenever hurricanes update
  useEffect(() => {
    if (hurricanes.length > 0) {
      try {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(hurricanes));
        localStorage.setItem(LOCAL_STORAGE_TIMESTAMP_KEY, Date.now().toString());
      } catch (err) {
        console.warn("Error caching hurricane data:", err);
      }
    }
  }, [hurricanes]);

  useEffect(() => {
    const loadPotentialStormAreas = async () => {
      const areas = await fetchPotentialStormAreas();
      setPotentialStormAreas(areas);
    };
    
    loadPotentialStormAreas();
    
    // Refresh the potential areas every 15 minutes
    const interval = setInterval(loadPotentialStormAreas, 15 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, []);
  

  // Fetch data from all configured sources
  async function fetchGlobalData(isBackgroundRefresh = false) {
    if (isDataFetching) return;
    
    try {
      if (!isBackgroundRefresh) {
        setLoading(true);
      }
      
      setIsDataFetching(true);
      
      console.log("Fetching global storm data...");
      // Collect hurricanes from all sources based on filter
      let allHurricanes = [];
      
      // Fetch multiple regions in parallel
      const fetchPromises = [];
      
      // US hurricanes from NOAA
      if (dataSource === 'all' || dataSource === 'us') {
        fetchPromises.push(
          getActiveHurricanes()
            .then(data => {
              console.log("NOAA data fetched:", data.length, "storms");
              return data;
            })
            .catch(error => {
              console.error("Error fetching NOAA data:", error);
              return [];
            })
        );
      }
      
      // Global regions using Open-Meteo
      const globalRegions = [];
      
      if (dataSource === 'all' || dataSource === 'japan') {
        globalRegions.push('WP'); // Western Pacific
      }
      
      if (dataSource === 'all' || dataSource === 'australia') {
        globalRegions.push('SP', 'SI'); // South Pacific, South Indian
      }
      
      if (dataSource === 'all') {
        // Add all other global regions
        globalRegions.push(
          'NI',       // North Indian
          'EP',       // Eastern Pacific  
          'EU',       // Europe
          'ASIA_WINTER',
          'EUROPE_WINTER',
          'SOUTH_AMERICA',
          'AFRICA'
        );
      }
      
      // Fetch Open-Meteo data for each region in parallel
      for (const region of globalRegions) {
        fetchPromises.push(
          getActiveHurricanesByRegion(region)
            .then(data => {
              console.log(`${region} data fetched:`, data.length, "storms");
              // Tag the data source
              return data.map(h => ({...h, dataSource: region}));
            })
            .catch(error => {
              console.error(`Error fetching ${region} data:`, error);
              return [];
            })
        );
      }
      
      // Wait for all fetches to complete
      const results = await Promise.all(fetchPromises);
      
      // Combine all results
      results.forEach(data => {
        if (data && data.length > 0) {
          allHurricanes = [...allHurricanes, ...data];
        }
      });
      
      // Filter out duplicate storms (by coordinates proximity)
      const uniqueHurricanes = filterDuplicateStorms(allHurricanes);
      
      console.log("Total unique storms found:", uniqueHurricanes.length);
      
      // Update the hurricane list
      setHurricanes(uniqueHurricanes);
      
      // Select first hurricane if available and none selected
      if (uniqueHurricanes.length > 0 && !selectedHurricane) {
        await handleHurricaneSelect(uniqueHurricanes[0]);
      } else if (selectedHurricane) {
        // If a hurricane is already selected, find it in the new data
        const updatedSelection = uniqueHurricanes.find(h => h.id === selectedHurricane.id);
        if (updatedSelection) {
          await handleHurricaneSelect(updatedSelection);
        }
      }
      
    } catch (err) {
      console.error("Error in fetchGlobalData:", err);
      setError(err.message);
    } finally {
      setLoading(false);
      setIsDataFetching(false);
    }
  }
  
  /**
 * Filter out duplicate storms based on proximity of coordinates
 * Preferring stronger storms and non-US data sources to balance the display
 */
  function filterDuplicateStorms(storms) {
    // First, ensure each storm has a truly unique ID by adding a suffix if needed
    storms = storms.map(storm => {
      if (!storm.id.includes('unique-')) {
        return {
          ...storm,
          id: `${storm.id}-unique-${Math.random().toString(36).substring(2, 8)}`
        };
      }
      return storm;
    });
    
    const uniqueStorms = [];
    const processedCoordinates = new Set();
    const stormIds = new Set(); // Track IDs to prevent duplicates
    
    // Sort storms by category strength (highest first) and prefer non-US sources
    const sortedStorms = [...storms].sort((a, b) => {
      // First sort by category
      const catA = getCategoryValue(a.category);
      const catB = getCategoryValue(b.category);
      
      if (catA !== catB) return catB - catA; // Higher category first
      
      // Then prefer non-US sources
      const isUSSourceA = !a.dataSource || a.dataSource === 'us';
      const isUSSourceB = !b.dataSource || b.dataSource === 'us';
      
      if (isUSSourceA !== isUSSourceB) {
        return isUSSourceA ? 1 : -1; // Non-US sources first
      }
      
      return 0;
    });
    
    // Filter based on coordinate proximity
    for (const storm of sortedStorms) {
      if (!storm.coordinates) continue;
      
      // Skip if ID already exists
      if (stormIds.has(storm.id)) continue;
      
      // Check if this storm is close to any already processed storm
      const stormKey = getCoordinateKey(storm.coordinates, 2.0); // 2 degree grid cells
      
      if (!processedCoordinates.has(stormKey)) {
        uniqueStorms.push(storm);
        processedCoordinates.add(stormKey);
        stormIds.add(storm.id);
      }
    }
    
    return uniqueStorms;
  }
  
  /**
   * Convert coordinates to a grid cell key for duplicate detection
   */
  function getCoordinateKey(coordinates, gridSize = 2.0) {
    if (!coordinates || coordinates.length < 2) return 'invalid';
    
    const [lon, lat] = coordinates;
    const gridLat = Math.floor(lat / gridSize) * gridSize;
    const gridLon = Math.floor(lon / gridSize) * gridSize;
    
    return `${gridLat},${gridLon}`;
  }
  
  /**
   * Convert category to numeric value for sorting
   */
  function getCategoryValue(category) {
    if (category === undefined || category === null) return -1;
    if (category === 'TD') return 0;
    if (category === 'TS') return 1;
    return parseInt(category) || -1;
  }

// Handle hurricane selection with Python backend integration
const handleHurricaneSelect = async (hurricane) => {
  if (!hurricane) {
    setSelectedHurricane(null);
    return;
  }
  
  setSelectedHurricane(hurricane);
  if (hurricane.coordinates) {
    try {
      // Call Python backend API to get all necessary storm data
      const stormData = await fetchStormDataFromPython(hurricane);
      
      // Update UI state with data from Python backend
      setObservations(stormData.observations);
      setForecastData(stormData.forecast);
      setRiskLevel(stormData.riskLevel);
      setSatelliteImagery(stormData.satelliteImagery);
      setHistoricalData(stormData.historicalData);
      
    } catch (err) {
      console.error("Error fetching storm data from Python backend:", err);
      
      // Fall back to JavaScript implementation
      try {
        let obs;
        // Existing JavaScript-based data fetching as fallback
        if (hurricane.dataSource === 'WP' || hurricane.dataSource === 'JAPAN') {
          obs = await getOpenMeteoObservations(
            hurricane.coordinates[1],
            hurricane.coordinates[0],
            'WP'
          );
        } else if (hurricane.dataSource === 'SP' || hurricane.dataSource === 'SI' || hurricane.dataSource === 'AUSTRALIA') {
          obs = await getOpenMeteoObservations(
            hurricane.coordinates[1],
            hurricane.coordinates[0],
            'SP'
          );
        } else {
          obs = await getHurricaneObservations(
            hurricane.coordinates[1],
            hurricane.coordinates[0]
          );
        }
        
        setObservations(obs);
        generateForecast(obs);
        calculateRiskLevel(hurricane, obs);
        fetchSatelliteImagery(hurricane.coordinates);
        generateHistoricalData();
      } catch (fallbackErr) {
        console.error("Fallback also failed:", fallbackErr);
      }
    }
  }
};

// New function to fetch storm data from Python backend
const fetchStormDataFromPython = async (hurricane) => {
  // Build request for the Python API
  const requestData = {
    coordinates: hurricane.coordinates,
    basin: hurricane.basin,
    hurricane_id: hurricane.id,
    name: hurricane.name,
    category: hurricane.category,
    data_source: hurricane.dataSource
  };

  console.log("Sending request to Python with data:", requestData);
  
  // Call Python API endpoint (you'll need to create this endpoint)
  const response = await fetch('http://localhost:8000/storm_data', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(requestData)
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch storm data: ${response.status}`);
  }
  
  return await response.json();
};

const fetchPotentialStormAreas = async () => {
  try {
    // Fetch potential storm formation areas from the Python backend
    const response = await fetch('http://localhost:8000/potential_storm_areas', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch potential storm areas: ${response.status}`);
    }
    
    const data = await response.json();
    return data.potential_areas || [];
  } catch (error) {
    console.error("Error fetching potential storm areas:", error);
    
    // Fallback to simulated data if the API fails
    return [
      // These are placeholders - your Python backend would generate the actual locations
      { id: 'pot1', position: [15.0, -40.0], probability: 0.7, basin: 'NA', intensity: 'TS' },
      { id: 'pot2', position: [12.0, 145.0], probability: 0.5, basin: 'WP', intensity: 'TD' },
      { id: 'pot3', position: [18.0, -110.0], probability: 0.4, basin: 'EP', intensity: 'TS' },
      { id: 'pot4', position: [-15.0, 85.0], probability: 0.3, basin: 'SI', intensity: 'TD' }
    ];
  }
};


  // Fetch satellite imagery for the selected hurricane
  const fetchSatelliteImagery = async (coordinates) => {
    if (!coordinates) return;
    
    try {
      // Simulate NASA imagery API call
      // Need to use the getNasaImagery function
      setSatelliteImagery({
        url: null, // Placeholder
        date: new Date().toISOString().split('T')[0],
        resolution: '250m'
      });
    } catch (err) {
      console.error("Error fetching satellite imagery:", err);
    }
  };

  // Generate forecast data
  const generateForecast = (obs) => {
    if (!obs) return;
    
    const forecast = [];
    let baseWind = obs.windSpeed || 75;
    let basePressure = obs.barometricPressure || 980;
    
    for (let hour = 0; hour <= 120; hour += 6) {
      // More realistic forecast with decreasing confidence
      const daysPast = hour / 24;
      const confidenceFactor = Math.max(0.5, 1 - (daysPast * 0.1));
      
      // Add some natural variability
      const windVariation = (Math.random() * 15 - 7.5) * (1 + daysPast * 0.2);
      const pressureVariation = (Math.random() * 8 - 4) * (1 + daysPast * 0.2);
      
      // Calculate uncertainty ranges
      const uncertaintyFactor = daysPast * 5;
      const windLow = baseWind + windVariation - uncertaintyFactor;
      const windHigh = baseWind + windVariation + uncertaintyFactor;
      
      // Create forecast point
      forecast.push({
        hour,
        day: Math.floor(hour / 24) + 1,
        windSpeed: baseWind + windVariation,
        windLow,
        windHigh,
        pressure: basePressure + pressureVariation,
        confidence: Math.max(20, Math.round(100 - (hour * 0.6))),
        category: getHurricaneCategory(baseWind + windVariation)
      });
      
      // Update base values with some persistence
      baseWind += (Math.random() * 10 - 5) * (1 + daysPast * 0.1);
      basePressure += (Math.random() * 4 - 2) * (1 + daysPast * 0.1);
    }
    
    setForecastData(forecast);
  };
  
  // Generate historical data (mock)
  const generateHistoricalData = () => {
    const history = [];
    const now = new Date();
    let baseWind = 65;
    let basePressure = 990;
    
    // Generate data for the past week
    for (let i = 168; i >= 0; i -= 6) {
      const timestamp = new Date(now);
      timestamp.setHours(timestamp.getHours() - i);
      
      // Add some natural variability with trend
      const trendFactor = 1 - i / 168; // Increasing trend (0 to 1)
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
    
    setHistoricalData(history);
  };

  // Get hurricane category based on wind speed
  const getHurricaneCategory = (windSpeed) => {
    if (windSpeed < 39) return 'TD';
    if (windSpeed < 74) return 'TS';
    if (windSpeed < 96) return '1';
    if (windSpeed < 111) return '2';
    if (windSpeed < 130) return '3';
    if (windSpeed < 157) return '4';
    return '5';
  };

  // Calculate risk level based on hurricane properties and observations
  const calculateRiskLevel = (hurricane, obs) => {
    if (!hurricane || !obs) return 'unknown';
    
    const factors = {
      category: hurricane.category || 0,
      windSpeed: obs.windSpeed || 0,
      pressure: obs.barometricPressure || 1013,
      rapidIntensification: false
    };
    
    if (obs.windSpeed > 100) {
      factors.rapidIntensification = true;
    }
    
    let riskScore = 0;
    riskScore += (factors.category * 15);
    riskScore += (factors.windSpeed / 2);
    riskScore += ((1013 - factors.pressure) / 2);
    if (factors.rapidIntensification) riskScore += 20;
    
    if (riskScore > 80) setRiskLevel('extreme');
    else if (riskScore > 60) setRiskLevel('high');
    else if (riskScore > 40) setRiskLevel('moderate');
    else setRiskLevel('low');
  };

  // Get risk color based on level
  const getRiskColor = (level) => {
    switch (level) {
      case 'extreme': return 'text-red-500';
      case 'high': return 'text-orange-500';
      case 'moderate': return 'text-yellow-500';
      default: return 'text-green-500';
    }
  };

  // Filter forecast data based on selected time period
  const getFilteredForecastData = () => {
    if (!forecastData.length) return [];
    
    switch (selectedTimePeriod) {
      case '24h': return forecastData.filter(d => d.hour <= 24);
      case '48h': return forecastData.filter(d => d.hour <= 48);
      case '72h': return forecastData.filter(d => d.hour <= 72);
      case '5d': return forecastData;
      default: return forecastData;
    }
  };

  // Get data for intensity chart
  const getIntensityChartData = () => {
    const filtered = getFilteredForecastData();
    
    // Add range info
    return filtered.map(data => ({
      ...data,
      range: [data.windLow, data.windHigh]
    }));
  };

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0B1021] text-white">
        <div className="text-xl">Loading storm data...</div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0B1021] text-white">
        <div className="text-xl text-red-500">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="w-full min-h-screen bg-[#0B1021] text-white">
      {/* Header */}
      <div className="w-full bg-[#1a237e] px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-yellow-400" />
          <span className="font-mono font-bold">
            Active Storms: {hurricanes.length}
          </span>
        </div>
        
        {/* Global Region Filter */}
        <div className="flex gap-2">
          <button 
            onClick={() => setDataSource('all')}
            className={`text-xs px-2 py-1 rounded-lg flex items-center gap-1 ${
              dataSource === 'all' ? 'bg-[#2a3890]' : 'bg-[#1a237e]/50 hover:bg-[#1a237e]'
            }`}
          >
            <Globe className="h-3 w-3" />
            <span>Global</span>
          </button>
          <button 
            onClick={() => setDataSource('us')}
            className={`text-xs px-2 py-1 rounded-lg ${
              dataSource === 'us' ? 'bg-[#2a3890]' : 'bg-[#1a237e]/50 hover:bg-[#1a237e]'
            }`}
          >
            US
          </button>
          <button 
            onClick={() => setDataSource('japan')}
            className={`text-xs px-2 py-1 rounded-lg ${
              dataSource === 'japan' ? 'bg-[#2a3890]' : 'bg-[#1a237e]/50 hover:bg-[#1a237e]'
            }`}
          >
            W Pacific
          </button>
          <button 
            onClick={() => setDataSource('australia')}
            className={`text-xs px-2 py-1 rounded-lg ${
              dataSource === 'australia' ? 'bg-[#2a3890]' : 'bg-[#1a237e]/50 hover:bg-[#1a237e]'
            }`}
          >
            Australia
          </button>
        </div>
        
        <div className="text-gray-300 text-sm">
          Last Updated: {new Date().toLocaleTimeString()}
        </div>
      </div>

      {/* Active Hurricanes List */}
  <div className="flex gap-2 mb-4 overflow-x-auto py-2 px-4 bg-[#0d1424]">
    {hurricanes.map(hurricane => (
      <button
        key={`button-${hurricane.id}`}
        onClick={() => handleHurricaneSelect(hurricane)}
        className={`px-4 py-2 rounded-lg transition-colors whitespace-nowrap flex items-center gap-1 ${
          selectedHurricane?.id === hurricane.id 
            ? 'bg-[#1a237e]' 
            : 'bg-[#1a237e]/50 hover:bg-[#1a237e]'
        }`}
      >
        <div 
          className="w-2 h-2 rounded-full" 
          style={{ 
            backgroundColor: hurricane.category >= 3 
              ? '#FF5050' 
              : hurricane.category >= 1 
                ? '#FFDE33' 
                : '#5DA5DA' 
          }}
        ></div>
        <span>{hurricane.name}</span>
        {hurricane.category > 0 && (
          <span className="text-xs bg-[#0d1424] px-1 rounded ml-1">
            Cat {hurricane.category}
          </span>
        )}
        {/* Show data source indicator */}
        {hurricane.dataSource && (
          <span className="text-xs text-gray-400 ml-1">
            {hurricane.dataSource === 'WP' || hurricane.dataSource === 'JAPAN' ? '(WP)' : 
            hurricane.dataSource === 'SP' || hurricane.dataSource === 'AUSTRALIA' ? '(AU)' : ''}
          </span>
        )}
      </button>
    ))}
  </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 p-4">
        {/* Map */}
        <div className="lg:col-span-2 bg-[#0d1424] rounded-lg overflow-hidden h-[500px] lg:h-[calc(100vh-200px)]">
          <AtlasCommandMap
            hurricanes={hurricanes}
            selectedHurricane={selectedHurricane}
            onSelectHurricane={handleHurricaneSelect}
            potentialStormAreas={potentialStormAreas}
          />
        </div>

        {/* Details Panel */}
        <div className="bg-[#0d1424] rounded-lg p-4 h-auto lg:h-[calc(100vh-200px)] overflow-y-auto">
          {selectedHurricane ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold">
                  {selectedHurricane.name}
                  {selectedHurricane.dataSource && (
                    <span className="text-sm font-normal text-gray-400 ml-2">
                      {selectedHurricane.dataSource === 'WP' || selectedHurricane.dataSource === 'JAPAN' ? 'Western Pacific' : 
                       selectedHurricane.dataSource === 'SP' || selectedHurricane.dataSource === 'SI' || selectedHurricane.dataSource === 'AUSTRALIA' ? 'South Pacific/Indian' : 
                       'Atlantic/Eastern Pacific'}
                    </span>
                  )}
                </h2>
                <div className="flex gap-2">
                  <button className="p-1 rounded hover:bg-[#1a237e]">
                    <Share className="h-4 w-4" />
                  </button>
                  <button className="p-1 rounded hover:bg-[#1a237e]">
                    <Download className="h-4 w-4" />
                  </button>
                </div>
              </div>
              
              {/* Current Conditions */}
              {observations && (
                <div className="bg-[#1a237e] p-4 rounded-lg space-y-2">
                  <h3 className="font-bold mb-2">Current Conditions</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="flex flex-col items-center">
                      <Wind className="h-5 w-5 text-blue-400 mb-1" />
                      <div className="text-lg font-bold">
                        {typeof observations.windSpeed === 'number' ? observations.windSpeed.toFixed(1) : 'N/A'}
                      </div>
                      <div className="text-xs text-gray-300">Wind (mph)</div>
                    </div>
                    <div className="flex flex-col items-center">
                      <ThermometerSun className="h-5 w-5 text-orange-400 mb-1" />
                      <div className="text-lg font-bold">
                        {typeof observations.temperature === 'number' ? observations.temperature.toFixed(1) : 'N/A'}
                      </div>
                      <div className="text-xs text-gray-300">Temp (°C)</div>
                    </div>
                    <div className="flex flex-col items-center">
                      <Droplets className="h-5 w-5 text-blue-300 mb-1" />
                      <div className="text-lg font-bold">
                        {typeof observations.relativeHumidity === 'number' ? observations.relativeHumidity.toFixed(0) : 'N/A'}
                      </div>
                      <div className="text-xs text-gray-300">Humidity (%)</div>
                    </div>
                    <div className="flex flex-col items-center">
                      <BarChart4 className="h-5 w-5 text-gray-300 mb-1" />
                      <div className="text-lg font-bold">
                        {typeof observations.barometricPressure === 'number' ? observations.barometricPressure.toFixed(0) : 'N/A'}
                      </div>
                      <div className="text-xs text-gray-300">Pressure (hPa)</div>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Enhanced Info Panel */}
              <div>
                <div className="flex border-b border-[#2a4858] mb-2">
                  <button 
                    onClick={() => setForecastView('track')}
                    className={`px-3 py-1 text-sm ${forecastView === 'track' ? 'border-b-2 border-blue-500' : 'text-gray-400'}`}
                  >
                    Forecast
                  </button>
                  <button 
                    onClick={() => setForecastView('intensity')}
                    className={`px-3 py-1 text-sm ${forecastView === 'intensity' ? 'border-b-2 border-blue-500' : 'text-gray-400'}`}
                  >
                    Intensity
                  </button>
                  <button 
                    onClick={() => setForecastView('risk')}
                    className={`px-3 py-1 text-sm ${forecastView === 'risk' ? 'border-b-2 border-blue-500' : 'text-gray-400'}`}
                  >
                    Risk
                  </button>
                </div>
                
                {/* Time Period Selector */}
                <div className="flex gap-1 mb-4">
                  {['24h', '48h', '72h', '5d'].map(period => (
                    <button
                      key={period}
                      onClick={() => setSelectedTimePeriod(period)}
                      className={`px-2 py-1 text-xs rounded ${
                        selectedTimePeriod === period 
                          ? 'bg-[#1a237e]' 
                          : 'bg-[#1a237e]/30'
                      }`}
                    >
                      {period}
                    </button>
                  ))}
                </div>
                
                {/* Content based on selected view */}
                {forecastView === 'track' && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold">Track Forecast</h4>
                    <div className="text-xs text-gray-300 mb-2">
                      Click on the map to see the detailed forecast path.
                    </div>
                    
                    {/* Forecast table */}
                    <div className="bg-[#1a237e]/30 rounded-lg overflow-hidden">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="bg-[#1a237e]">
                            <th className="py-2 px-2 text-left">Time</th>
                            <th className="py-2 px-2 text-center">Position</th>
                            <th className="py-2 px-2 text-right">Category</th>
                          </tr>
                          </thead>
                        <tbody>
                          {getFilteredForecastData()
                            .filter(d => d.hour % 24 === 0)
                            .map((point, index) => (
                              <tr key={index} className="border-t border-[#2a4858]">
                                <td className="py-2 px-2">+{point.hour}h</td>
                                <td className="py-2 px-2 text-center">
                                  {selectedHurricane.coordinates ? 
                                    `${(selectedHurricane.coordinates[1] + index * 0.5).toFixed(1)}°N, ${(selectedHurricane.coordinates[0] + index * 0.7).toFixed(1)}°W` : 
                                    'N/A'
                                  }
                                </td>
                                <td className="py-2 px-2 text-right">
                                  <span className={`px-1.5 py-0.5 rounded text-xs ${
                                    point.category === '5' ? 'bg-red-500/30' :
                                    point.category === '4' ? 'bg-orange-500/30' :
                                    point.category === '3' ? 'bg-yellow-500/30' :
                                    point.category === '2' ? 'bg-cyan-500/30' :
                                    point.category === '1' ? 'bg-blue-500/30' :
                                    'bg-blue-300/30'
                                  }`}>
                                    {point.category}
                                  </span>
                                </td>
                              </tr>
                            ))
                          }
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                
                {forecastView === 'intensity' && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold">Intensity Forecast</h4>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={getIntensityChartData()}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
                          <XAxis 
                            dataKey="hour" 
                            label={{ value: 'Hours', position: 'insideBottom', offset: -5 }}
                            stroke="#fff" 
                          />
                          <YAxis 
                            label={{ value: 'Wind Speed (mph)', angle: -90, position: 'insideLeft' }} 
                            stroke="#fff" 
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#1a237e',
                              border: 'none',
                              borderRadius: '4px'
                            }}
                            formatter={(value) => [`${Math.round(value)} mph`, 'Wind Speed']}
                          />
                          <Line
                            type="monotone"
                            dataKey="windSpeed"
                            stroke="#4f46e5"
                            strokeWidth={2}
                            dot={false}
                          />
                          {/* Uncertainty range as an area */}
                          <Area
                            type="monotone"
                            dataKey="range"
                            fill="#4f46e5"
                            fillOpacity={0.2}
                            stroke="none"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    
                    {/* Category thresholds */}
                    <div className="flex justify-between text-xs mt-2">
                      <div className="flex items-center">
                        <div className="w-2 h-2 rounded-full bg-[#5DA5DA] mr-1"></div>
                        <span>TS: 39mph</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 rounded-full bg-[#4DC4FF] mr-1"></div>
                        <span>Cat 1: 74mph</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 rounded-full bg-[#FFDE33] mr-1"></div>
                        <span>Cat 3: 111mph</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 rounded-full bg-[#FF5050] mr-1"></div>
                        <span>Cat 5: 157mph</span>
                      </div>
                    </div>
                  </div>
                )}
                
                {forecastView === 'risk' && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-bold">Risk Assessment</h4>
                    
                    {/* Overall risk */}
                    <div className={`text-lg font-bold ${getRiskColor(riskLevel)} flex items-center justify-between bg-[#1a237e]/30 p-4 rounded-lg`}>
                      <span>{riskLevel.toUpperCase()} RISK</span>
                      <div className="text-sm font-normal">Confidence: 80%</div>
                    </div>
                    
                    {/* Risk details */}
                    <div className="grid grid-cols-2 gap-2 mt-4">
                      <div className="bg-[#1a237e]/30 p-3 rounded-lg">
                        <div className="text-xs text-gray-300">Wind Threat</div>
                        <div className={`text-lg font-bold ${observations?.windSpeed > 100 ? 'text-red-500' : 'text-yellow-500'}`}>
                          {observations?.windSpeed > 100 ? 'Severe' : 'Moderate'}
                        </div>
                      </div>
                      <div className="bg-[#1a237e]/30 p-3 rounded-lg">
                        <div className="text-xs text-gray-300">Storm Surge</div>
                        <div className={`text-lg font-bold ${selectedHurricane.category > 2 ? 'text-red-500' : 'text-yellow-500'}`}>
                          {selectedHurricane.category > 2 ? 'High' : 'Moderate'}
                        </div>
                      </div>
                      <div className="bg-[#1a237e]/30 p-3 rounded-lg">
                        <div className="text-xs text-gray-300">Flooding Risk</div>
                        <div className={`text-lg font-bold text-yellow-500`}>
                          Moderate
                        </div>
                      </div>
                      <div className="bg-[#1a237e]/30 p-3 rounded-lg">
                        <div className="text-xs text-gray-300">Rapid Intensification</div>
                        <div className={`text-lg font-bold ${observations?.windSpeed > 100 ? 'text-red-500' : 'text-green-500'}`}>
                          {observations?.windSpeed > 100 ? 'Likely' : 'Unlikely'}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="text-gray-400 mb-4">
                <AlertTriangle className="h-12 w-12 mx-auto mb-2" />
                <p>Select a storm to view details</p>
              </div>
              <p className="text-sm text-gray-500">
                Click on any storm in the list above or directly on the map
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}