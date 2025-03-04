'use client';

import dynamic from 'next/dynamic';
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Wind, Droplets, Navigation, ThermometerSun } from 'lucide-react';
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

  const handleHurricaneSelect = async (hurricane) => {
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
      } catch (err) {
        console.error("Error fetching observations:", err);
      }
    }
  };

  const generateForecast = (obs) => {
    if (!obs) return;
    
    const forecast = [];
    let baseWind = obs.windSpeed || 75;
    let basePressure = obs.barometricPressure || 980;
    
    for (let hour = 0; hour <= 24; hour += 3) {
      const windVariation = Math.random() * 10 - 5;
      const pressureVariation = Math.random() * 5 - 2.5;
      
      forecast.push({
        hour,
        windSpeed: baseWind + windVariation,
        pressure: basePressure + pressureVariation,
        confidence: Math.max(100 - (hour * 3), 50) // Confidence decreases over time
      });
    }
    
    setForecastData(forecast);
  };

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

  const getPredictedPath = (hurricane) => {
    if (!hurricane.coordinates) return [];
    
    const [lon, lat] = hurricane.coordinates;
    const positions = [[lat, lon]];
    
    for (let i = 1; i <= 24; i += 6) {
      positions.push([
        lat + (i * 0.1),
        lon + (i * 0.15)
      ]);
    }
    
    return positions;
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'extreme': return 'text-red-500';
      case 'high': return 'text-orange-500';
      case 'moderate': return 'text-yellow-500';
      default: return 'text-green-500';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0B1021] text-white">
        <div className="text-xl">Loading hurricane data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0B1021] text-white">
        <div className="text-xl text-red-500">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="w-full h-screen bg-[#0B1021] text-white p-6">
      {/* Header */}
      <div className="w-full bg-[#1a237e] px-4 py-2 rounded-lg mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-400" />
            <span className="font-mono font-bold">
              Active Hurricanes: {hurricanes.length}
            </span>
          </div>
        </div>
      </div>

      {/* Active Hurricanes List */}
      <div className="flex gap-4 mb-4 overflow-x-auto pb-2">
        {hurricanes.map(hurricane => (
          <button
            key={hurricane.id}
            onClick={() => handleHurricaneSelect(hurricane)}
            className={`px-4 py-2 rounded-lg transition-colors whitespace-nowrap ${
              selectedHurricane?.id === hurricane.id 
                ? 'bg-[#1a237e]' 
                : 'bg-[#1a237e]/50 hover:bg-[#1a237e]'
            }`}
          >
            {hurricane.name} {hurricane.category > 0 && `(Cat ${hurricane.category})`}
          </button>
        ))}
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-3 gap-4">
        {/* Map */}
        <div className="col-span-2 bg-[#0d1424] rounded-lg overflow-hidden h-[calc(100vh-250px)]">
          <AtlasCommandMap
            hurricanes={hurricanes}
            selectedHurricane={selectedHurricane}
            onSelectHurricane={handleHurricaneSelect}
          />
        </div>

        {/* Details Panel */}
        <div className="bg-[#0d1424] rounded-lg p-4 h-[calc(100vh-250px)] overflow-y-auto">
          {selectedHurricane ? (
            <div className="space-y-4">
              <h2 className="text-xl font-bold">{selectedHurricane.name}</h2>
              
              {/* Current Conditions */}
              {observations && (
                <div className="bg-[#1a237e] p-4 rounded-lg space-y-2">
                  <h3 className="font-bold mb-2">Current Conditions</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="flex items-center gap-2">
                      <Wind className="h-4 w-4 text-blue-400" />
                      <span>{observations.windSpeed} mph</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <ThermometerSun className="h-4 w-4 text-orange-400" />
                      <span>{observations.temperature}°C</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Droplets className="h-4 w-4 text-blue-300" />
                      <span>{observations.relativeHumidity}%</span>
                    </div>
                  </div>
                </div>
              )}

              {/* AI Prediction Component */}
              <HurricanePrediction 
                selectedHurricane={selectedHurricane} 
                nasaService={{ getNasaImagery, getGibsLayers }}
              />

              {/* Intensity Forecast */}
              <div className="bg-[#1a237e] p-4 rounded-lg">
                <h3 className="font-bold mb-4">24-Hour Intensity Forecast</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={forecastData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
                      <XAxis dataKey="hour" stroke="#fff" />
                      <YAxis stroke="#fff" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1a237e',
                          border: 'none',
                          borderRadius: '4px'
                        }}
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
              </div>

              {/* Risk Assessment */}
              <div className="bg-[#1a237e] p-4 rounded-lg">
                <h3 className="font-bold mb-2">Risk Assessment</h3>
                <div className={`text-lg font-bold ${getRiskColor(riskLevel)}`}>
                  {riskLevel.toUpperCase()} RISK
                </div>
                <div className="mt-4 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Wind Threat:</span>
                    <span className={getRiskColor(riskLevel)}>
                      {observations?.windSpeed > 100 ? 'Severe' : 'Moderate'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Storm Surge Potential:</span>
                    <span className={getRiskColor(riskLevel)}>
                      {selectedHurricane.category > 2 ? 'High' : 'Moderate'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Rapid Intensification:</span>
                    <span className={getRiskColor(riskLevel)}>
                      {observations?.windSpeed > 100 ? 'Likely' : 'Unlikely'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Alert Details */}
              <div className="bg-[#1a237e] p-4 rounded-lg">
                <h3 className="font-bold mb-2">Alert Details</h3>
                <div className="space-y-2 text-sm">
                  <p><span className="text-gray-400">Severity:</span> {selectedHurricane.severity}</p>
                  <p><span className="text-gray-400">Certainty:</span> {selectedHurricane.certainty}</p>
                  <p><span className="text-gray-400">Areas:</span> {selectedHurricane.areas}</p>
                </div>
              </div>

              {/* Instructions */}
              {selectedHurricane.instruction && (
                <div className="bg-[#1a237e] p-4 rounded-lg">
                  <h3 className="font-bold mb-2">Instructions</h3>
                  <p className="text-sm">{selectedHurricane.instruction}</p>
                </div>
              )}

              {/* Description */}
              {selectedHurricane.description && (
                <div className="bg-[#1a237e] p-4 rounded-lg">
                  <h3 className="font-bold mb-2">Description</h3>
                  <p className="text-sm">{selectedHurricane.description}</p>
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-400">Select a hurricane to view details</p>
          )}
        </div>
      </div>
    </div>
  );
}