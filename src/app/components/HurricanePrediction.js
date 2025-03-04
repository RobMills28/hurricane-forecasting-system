// components/HurricanePrediction.js
'use client';

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { HurricaneEnvironment, HurricanePredictionAgent } from '../hurricaneAgent';
import { fetchHistoricalHurricaneData, preprocessDataForTraining } from '../hurricaneData';

const HurricanePrediction = ({ selectedHurricane, nasaService }) => {
  const [agent, setAgent] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [predictions, setPredictions] = useState([]);
  const [performance, setPerformance] = useState(null);

  // Initialize agent when component mounts
  useEffect(() => {
    async function initializeAgent() {
      try {
        setTrainingStatus('loading');
        
        // Fetch historical data
        const historicalData = await fetchHistoricalHurricaneData();
        const processedData = preprocessDataForTraining(historicalData);
        
        // Create environment and agent
        const environment = new HurricaneEnvironment();
        await environment.initialize(processedData, nasaService);
        
        const newAgent = new HurricanePredictionAgent();
        setAgent(newAgent);
        
        // Train agent
        setTrainingStatus('training');
        const episodes = 20;
        
        // Mock progress updates during training
        const progressInterval = setInterval(() => {
          setTrainingProgress(prev => {
            if (prev >= 95) {
              clearInterval(progressInterval);
              return prev;
            }
            return prev + 5;
          });
        }, 300);
        
        const trainingPerformance = await newAgent.train(environment, episodes);
        clearInterval(progressInterval);
        setTrainingProgress(100);
        setTrainingStatus('ready');
        setPerformance({
          positionError: trainingPerformance.map(p => p.avgPosError),
          intensityError: trainingPerformance.map(p => p.avgIntensityError)
        });
        
      } catch (error) {
        console.error('Error initializing prediction agent:', error);
        setTrainingStatus('error');
      }
    }
    
    initializeAgent();
  }, [nasaService]);

  // Generate predictions when selected hurricane changes
  useEffect(() => {
    if (!agent || !selectedHurricane || trainingStatus !== 'ready') return;
    
    // Create prediction for the selected hurricane
    const currentState = {
      position: {
        lat: selectedHurricane.coordinates ? selectedHurricane.coordinates[1] : 0,
        lon: selectedHurricane.coordinates ? selectedHurricane.coordinates[0] : 0
      },
      windSpeed: selectedHurricane.windSpeed || 75,
      pressure: selectedHurricane.pressure || 980,
      seaSurfaceTemp: { value: 28 }, // Default if no NASA data available
      timestamp: new Date()
    };
    
    // Generate forecasts for the next 5 days (at 24-hour intervals)
    const forecastDays = 5;
    const newPredictions = [];
    let predictState = {...currentState};
    let history = [];
    
    for (let day = 1; day <= forecastDays; day++) {
      const prediction = agent.predict(predictState, history);
      const timestamp = new Date();
      timestamp.setDate(timestamp.getDate() + day);
      
      newPredictions.push({
        day,
        timestamp,
        position: prediction.position,
        windSpeed: prediction.windSpeed,
        category: getHurricaneCategory(prediction.windSpeed)
      });
      
      // Update for next prediction
      history.push({ state: predictState, action: prediction });
      predictState = {
        ...predictState,
        position: prediction.position,
        windSpeed: prediction.windSpeed
      };
    }
    
    setPredictions(newPredictions);
  }, [agent, selectedHurricane, trainingStatus]);

  // Helper to get hurricane category from wind speed
  const getHurricaneCategory = (windSpeed) => {
    if (windSpeed < 74) return 'TS';
    if (windSpeed < 96) return '1';
    if (windSpeed < 111) return '2';
    if (windSpeed < 130) return '3';
    if (windSpeed < 157) return '4';
    return '5';
  };

  return (
    <div className="bg-[#1a237e] p-4 rounded-lg">
      <h3 className="font-bold mb-2">AI-Powered Forecast</h3>
      
      {trainingStatus === 'loading' && (
        <div className="text-center py-4">
          <p>Loading historical hurricane data...</p>
        </div>
      )}
      
      {trainingStatus === 'training' && (
        <div className="text-center py-4">
          <p>Training prediction model... {trainingProgress}%</p>
          <div className="w-full bg-[#0B1021] h-2 mt-2 rounded-full overflow-hidden">
            <div 
              className="bg-blue-500 h-full" 
              style={{ width: `${trainingProgress}%` }}
            ></div>
          </div>
        </div>
      )}
      
      {trainingStatus === 'error' && (
        <div className="text-center py-4 text-red-400">
          <p>Error training prediction model. Please try again.</p>
        </div>
      )}
      
      {trainingStatus === 'ready' && predictions.length > 0 && (
        <>
          <div className="mb-4">
            <h4 className="text-sm font-bold mb-2">5-Day Forecast</h4>
            <div className="space-y-2 text-sm">
              {predictions.map(pred => (
                <div key={pred.day} className="flex justify-between items-center">
                  <span>Day {pred.day}:</span>
                  <div className="flex gap-4">
                    <span className="text-gray-300">
                      {pred.position.lat.toFixed(1)}°N, {Math.abs(pred.position.lon).toFixed(1)}°W
                    </span>
                    <span>
                      Cat {pred.category} ({Math.round(pred.windSpeed)} mph)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="h-48">
            <h4 className="text-sm font-bold mb-2">Intensity Forecast</h4>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={predictions.map(p => ({ day: p.day, windSpeed: p.windSpeed }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
                <XAxis dataKey="day" label={{ value: 'Days', position: 'insideBottom', dy: 10 }} stroke="#fff" />
                <YAxis label={{ value: 'Wind Speed (mph)', angle: -90, position: 'insideLeft' }} stroke="#fff" />
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
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          {performance && (
            <div className="mt-4 text-xs text-gray-400">
              <p>Model trained on {performance.positionError.length} episodes</p>
              <p>Avg. Position Error: {performance.positionError[performance.positionError.length - 1].toFixed(2)} km</p>
              <p>Avg. Intensity Error: {performance.intensityError[performance.intensityError.length - 1].toFixed(2)} mph</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default HurricanePrediction;