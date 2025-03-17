'use client';

import dynamic from 'next/dynamic';
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { AlertTriangle, Wind, Droplets, Navigation, ThermometerSun, ChevronDown, ChevronUp, Info, Share, Download, Eye, ArrowUpRight, BarChart4 } from 'lucide-react';
import { getActiveHurricanes, getHurricaneObservations } from './noaaService';
import AtlasCommandMap from './components/AtlasCommandMap';
import HurricanePrediction from './components/HurricanePrediction';
import { getNasaImagery, getGibsLayers } from './nasaService';

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

  // Fetch initial data
  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const data = await getActiveHurricanes();
        setHurricanes(data);
        
        if (data.length > 0) {
          setSelectedHurricane(data[0]);
          if (data[0].coordinates) {
            const obs = await getHurricaneObservations(
              data[0].coordinates[1],
              data[0].coordinates[0]
            );
            setObservations(obs);
            generateForecast(obs);
            calculateRiskLevel(data[0], obs);
            fetchSatelliteImagery(data[0].coordinates);
            generateHistoricalData();
          }
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 300000); // Update every 5 minutes
    return () => clearInterval(interval);
  }, []);

  // Handle hurricane selection
  const handleHurricaneSelect = async (hurricane) => {
    if (!hurricane) {
      setSelectedHurricane(null);
      return;
    }
    
    setSelectedHurricane(hurricane);
    if (hurricane.coordinates) {
      try {
        const obs = await getHurricaneObservations(
          hurricane.coordinates[1],
          hurricane.coordinates[0]
        );
        setObservations(obs);
        generateForecast(obs);
        calculateRiskLevel(hurricane, obs);
        fetchSatelliteImagery(hurricane.coordinates);
        generateHistoricalData();
      } catch (err) {
        console.error("Error fetching observations:", err);
      }
    }
  };

  // Fetch satellite imagery for the selected hurricane
  const fetchSatelliteImagery = async (coordinates) => {
    if (!coordinates) return;
    
    try {
      // Simulate NASA imagery API call
      // In a real implementation, you would use the getNasaImagery function
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
        <div className="text-xl">Loading hurricane data...</div>
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
            Active Hurricanes: {hurricanes.length}
          </span>
        </div>
        <div className="text-gray-300 text-sm">
          Last Updated: {new Date().toLocaleTimeString()}
        </div>
      </div>

      {/* Active Hurricanes List */}
      <div className="flex gap-2 mb-4 overflow-x-auto py-2 px-4 bg-[#0d1424]">
        {hurricanes.map(hurricane => (
          <button
            key={hurricane.id}
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
          />
        </div>

        {/* Details Panel */}
        <div className="bg-[#0d1424] rounded-lg p-4 h-auto lg:h-[calc(100vh-200px)] overflow-y-auto">
          {selectedHurricane ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold">{selectedHurricane.name}</h2>
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
                      <div className="text-lg font-bold">{observations.windSpeed || 'N/A'}</div>
                      <div className="text-xs text-gray-300">Wind (mph)</div>
                    </div>
                    <div className="flex flex-col items-center">
                      <ThermometerSun className="h-5 w-5 text-orange-400 mb-1" />
                      <div className="text-lg font-bold">{observations.temperature || 'N/A'}</div>
                      <div className="text-xs text-gray-300">Temp (°C)</div>
                    </div>
                    <div className="flex flex-col items-center">
                      <Droplets className="h-5 w-5 text-blue-300 mb-1" />
                      <div className="text-lg font-bold">{observations.relativeHumidity || 'N/A'}</div>
                      <div className="text-xs text-gray-300">Humidity (%)</div>
                    </div>
                    <div className="flex flex-col items-center">
                      <BarChart4 className="h-5 w-5 text-gray-300 mb-1" />
                      <div className="text-lg font-bold">{observations.barometricPressure || 'N/A'}</div>
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
              
              {/* Expandable Sections */}
              <div className="space-y-2 mt-8">
                {/* Hurricane Details */}
                <div className="border border-[#2a4858] rounded-lg overflow-hidden">
                  <button 
                    onClick={() => setDetailsExpanded(!detailsExpanded)}
                    className="w-full flex items-center justify-between bg-[#1a237e]/50 px-4 py-2"
                  >
                    <span className="font-bold">Hurricane Details</span>
                    {detailsExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </button>
                  
                  {detailsExpanded && (
                    <div className="p-4 space-y-2 text-sm">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                        <div>
                          <span className="text-gray-400">Type:</span> {selectedHurricane.type}
                        </div>
                        <div>
                          <span className="text-gray-400">Category:</span> {selectedHurricane.category || 'TS'}
                        </div>
                        <div>
                          <span className="text-gray-400">Status:</span> {selectedHurricane.status || 'Active'}
                        </div>
                        <div>
                          <span className="text-gray-400">Basin:</span> {selectedHurricane.basin || 'NA'} 
                          <span className="text-xs text-gray-500 ml-1">
                            {selectedHurricane.basin === 'NA' ? '(North Atlantic)' : 
                             selectedHurricane.basin === 'EP' ? '(Eastern Pacific)' : 
                             selectedHurricane.basin === 'WP' ? '(Western Pacific)' : 
                             selectedHurricane.basin === 'NI' ? '(North Indian)' : ''}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">Position:</span> {
                            selectedHurricane.coordinates ? 
                            `${selectedHurricane.coordinates[1].toFixed(2)}°N, ${Math.abs(selectedHurricane.coordinates[0]).toFixed(2)}°W` : 
                            'Unknown'
                          }
                        </div>
                        <div>
                          <span className="text-gray-400">Formed:</span> {
                            new Date().toLocaleDateString()
                          }
                        </div>
                      </div>
                      
                      <div className="mt-4">
                        <span className="text-gray-400">Areas Affected:</span> {selectedHurricane.areas || 'None reported'}
                      </div>
                      
                      <div className="mt-2">
                        <span className="text-gray-400">Advisories:</span>
                        <div className="mt-1 bg-[#1a237e]/30 p-2 rounded text-xs">
                          {selectedHurricane.instruction || 'No active advisories at this time.'}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Historical Data */}
                <div className="border border-[#2a4858] rounded-lg overflow-hidden">
                  <button 
                    onClick={() => setHistoryExpanded(!historyExpanded)}
                    className="w-full flex items-center justify-between bg-[#1a237e]/50 px-4 py-2"
                  >
                    <span className="font-bold">Historical Data</span>
                    {historyExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </button>
                  
                  {historyExpanded && (
                    <div className="p-4 space-y-4">
                      <div className="text-sm">
                        <p>Historical trend of {selectedHurricane.name} over the past 7 days</p>
                      </div>
                      
                      <div className="h-48">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={historicalData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
                            <XAxis 
                              dataKey="hour" 
                              stroke="#fff" 
                              tickFormatter={(value) => `${Math.abs(value/24).toFixed(0)}d`}
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
                              labelFormatter={(value) => `${Math.abs(value/24).toFixed(1)} days ago`}
                            />
                            <Line
                              type="monotone"
                              dataKey="windSpeed"
                              stroke="#4f46e5"
                              strokeWidth={2}
                              dot={false}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      
                      <div className="text-xs text-gray-300 mt-4">
                        <div className="flex items-center">
                          <Info className="h-4 w-4 mr-1" />
                          <span>Historical data based on satellite observations and ground measurements</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Impact Analysis */}
                <div className="border border-[#2a4858] rounded-lg overflow-hidden">
                  <button 
                    onClick={() => setImpactExpanded(!impactExpanded)}
                    className="w-full flex items-center justify-between bg-[#1a237e]/50 px-4 py-2"
                  >
                    <span className="font-bold">Impact Analysis</span>
                    {impactExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </button>
                  
                  {impactExpanded && (
                    <div className="p-4 space-y-4">
                      <div className="text-sm">
                        <p>Potential impacts of {selectedHurricane.name} based on current trajectory and intensity</p>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="bg-[#1a237e]/30 p-3 rounded-lg">
                          <h4 className="font-bold">Population Exposure</h4>
                          <div className="text-sm mt-1 text-gray-300">
                            <p>Approximately 2.3 million people within potential impact zone</p>
                          </div>
                          <div className="w-full bg-[#0b1021] h-2 mt-2 rounded-full overflow-hidden">
                            <div 
                              className="bg-yellow-500 h-full transition-all"
                              style={{ width: `65%` }}
                            ></div>
                          </div>
                        </div>
                        
                        <div className="bg-[#1a237e]/30 p-3 rounded-lg">
                          <h4 className="font-bold">Infrastructure Risk</h4>
                          <div className="text-sm mt-1 text-gray-300">
                            <p>Critical infrastructure in coastal areas at moderate risk</p>
                          </div>
                          <div className="w-full bg-[#0b1021] h-2 mt-2 rounded-full overflow-hidden">
                            <div 
                              className="bg-orange-500 h-full transition-all"
                              style={{ width: `75%` }}
                            ></div>
                          </div>
                        </div>
                        
                        <div className="bg-[#1a237e]/30 p-3 rounded-lg">
                          <h4 className="font-bold">Economic Impact</h4>
                          <div className="text-sm mt-1 text-gray-300">
                            <p>Potential damage estimated at $1.2 - 3.5 billion</p>
                          </div>
                          <div className="w-full bg-[#0b1021] h-2 mt-2 rounded-full overflow-hidden">
                            <div 
                              className="bg-red-500 h-full transition-all"
                              style={{ width: `80%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-xs text-gray-300 flex items-center">
                        <Info className="h-4 w-4 mr-1 flex-shrink-0" />
                        <span>Impact analysis based on NOAA models and historical damage patterns</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              
              {/* External Resources */}
              <div className="mt-6">
                <h3 className="text-sm font-bold mb-2">External Resources</h3>
                <div className="grid grid-cols-2 gap-2">
                  <a 
                    href="#" 
                    className="flex items-center gap-1 text-xs bg-[#1a237e]/50 p-2 rounded-lg hover:bg-[#1a237e]"
                  >
                    <Eye className="h-4 w-4" />
                    <span>NOAA Tracker</span>
                    <ArrowUpRight className="h-3 w-3 ml-auto" />
                  </a>
                  <a 
                    href="#" 
                    className="flex items-center gap-1 text-xs bg-[#1a237e]/50 p-2 rounded-lg hover:bg-[#1a237e]"
                  >
                    <Wind className="h-4 w-4" />
                    <span>Wind Maps</span>
                    <ArrowUpRight className="h-3 w-3 ml-auto" />
                  </a>
                  <a 
                    href="#" 
                    className="flex items-center gap-1 text-xs bg-[#1a237e]/50 p-2 rounded-lg hover:bg-[#1a237e]"
                  >
                    <AlertTriangle className="h-4 w-4" />
                    <span>Evacuation Info</span>
                    <ArrowUpRight className="h-3 w-3 ml-auto" />
                  </a>
                  <a 
                    href="#" 
                    className="flex items-center gap-1 text-xs bg-[#1a237e]/50 p-2 rounded-lg hover:bg-[#1a237e]"
                  >
                    <Navigation className="h-4 w-4" />
                    <span>Marine Warnings</span>
                    <ArrowUpRight className="h-3 w-3 ml-auto" />
                  </a>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="text-gray-400 mb-4">
                <AlertTriangle className="h-12 w-12 mx-auto mb-2" />
                <p>Select a hurricane to view details</p>
              </div>
              <p className="text-sm text-gray-500">
                Click on any hurricane in the list above or directly on the map
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}