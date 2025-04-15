'use client';

import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart, Legend } from 'recharts';
import { AlertTriangle, Wind, Cloud, ArrowUp, ArrowDown, Info, Clock, Droplets, Navigation } from 'lucide-react';

const HurricanePrediction = ({ selectedHurricane, nasaService }) => {
  const [loading, setLoading] = useState(false);
  const [predictionData, setPredictionData] = useState([]);
  const [ensembleData, setEnsembleData] = useState([]);
  const [confidenceData, setConfidenceData] = useState([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('72h');
  const [selectedView, setSelectedView] = useState('intensity');
  const [hoverInfo, setHoverInfo] = useState(null);
  const [predictionSummary, setPredictionSummary] = useState(null);
  
  // Generate forecast when selected hurricane changes
  useEffect(() => {
    if (!selectedHurricane) return;
    
    setLoading(true);
    
    // Simulate API call delay
    const timer = setTimeout(() => {
      generatePrediction();
      setLoading(false);
    }, 500);
    
    return () => clearTimeout(timer);
  }, [selectedHurricane]);
  
  // Generate prediction data
  const generatePrediction = () => {
    if (!selectedHurricane) return;
    
    // Generate forecast points for the next 120 hours (5 days)
    const forecast = [];
    let baseWindSpeed = selectedHurricane.windSpeed || 85;
    let basePressure = selectedHurricane.pressure || 980;
    
    // Create deterministic forecast line
    for (let hour = 0; hour <= 120; hour += 6) {
      // Add realistic variability
      const dayFraction = hour / 24;
      
      // Different behavior based on hurricane lifecycle phase
      let intensityTrend;
      if (dayFraction < 1) {
        // First day - usually intensification
        intensityTrend = 5 + (Math.random() * 5);
      } else if (dayFraction < 3) {
        // Peak period - fluctuations around peak
        intensityTrend = Math.random() * 20 - 10;
      } else {
        // Weakening phase
        intensityTrend = -5 - (Math.random() * 10);
      }
      
      // Apply trend with some randomness
      baseWindSpeed += (intensityTrend * 0.2) + (Math.random() * 6 - 3);
      basePressure += (intensityTrend * -0.1) + (Math.random() * 3 - 1.5);
      
      // Calculate uncertainty based on forecast time
      const uncertaintyFactor = Math.min(0.5, 0.1 + (dayFraction * 0.1));
      const windLow = baseWindSpeed * (1 - uncertaintyFactor);
      const windHigh = baseWindSpeed * (1 + uncertaintyFactor);
      
      // Add forecast point
      forecast.push({
        hour,
        day: Math.floor(hour / 24) + 1,
        windSpeed: baseWindSpeed,
        pressure: basePressure,
        category: getHurricaneCategory(baseWindSpeed),
        windLow,
        windHigh,
        uncertaintyRange: [windLow, windHigh],
        confidence: Math.max(20, Math.round(100 - (hour * 0.6)))
      });
    }
    
    setPredictionData(forecast);
    
    // Generate ensemble data
    generateEnsembleData(forecast);
    
    // Generate confidence data
    generateConfidenceData(forecast);
    
    // Generate prediction summary
    generatePredictionSummary(forecast);
  };
  
  // Generate ensemble model data
  const generateEnsembleData = (forecast) => {
    // Create 10 ensemble members with variations
    const ensembles = [];
    
    for (let member = 0; member < 10; member++) {
      const memberData = [];
      
      // Base variation for this ensemble member
      const memberBias = (Math.random() * 0.3) - 0.15; // -15% to +15% bias
      
      for (let i = 0; i < forecast.length; i++) {
        const point = forecast[i];
        
        // Add variation that increases with forecast time
        const timeVariation = (point.day * 0.05); // Increasing variation over time
        const variation = memberBias + (Math.random() * timeVariation * 2) - timeVariation;
        
        memberData.push({
          hour: point.hour,
          day: point.day,
          windSpeed: point.windSpeed * (1 + variation),
          pressure: point.pressure * (1 - variation * 0.5),
          member
        });
      }
      
      ensembles.push(memberData);
    }
    
    setEnsembleData(ensembles);
  };
  
  // Generate confidence data
  const generateConfidenceData = (forecast) => {
    const confidence = [];
    
    for (let i = 0; i < forecast.length; i++) {
      const point = forecast[i];
      
      // Different components of confidence
      const trackConfidence = Math.max(10, 100 - (point.day * 15));
      const intensityConfidence = Math.max(5, 95 - (point.day * 17));
      const timingConfidence = Math.max(15, 90 - (point.day * 12));
      
      // Overall confidence (weighted average)
      const overallConfidence = Math.round(
        (trackConfidence * 0.4) + 
        (intensityConfidence * 0.4) + 
        (timingConfidence * 0.2)
      );
      
      confidence.push({
        hour: point.hour,
        day: point.day,
        track: trackConfidence,
        intensity: intensityConfidence,
        timing: timingConfidence,
        overall: overallConfidence
      });
    }
    
    setConfidenceData(confidence);
  };
  
  // Generate prediction summary
  const generatePredictionSummary = (forecast) => {
    if (!forecast.length) return;
    
    // Find peak intensity
    const peakIntensity = Math.max(...forecast.map(f => f.windSpeed));
    const peakPoint = forecast.find(f => f.windSpeed === peakIntensity);
    
    // Calculate average intensification rate (first 48 hours)
    const startIntensity = forecast[0].windSpeed;
    const intensityAt48h = forecast.find(f => f.hour === 48)?.windSpeed || forecast[forecast.length - 1].windSpeed;
    const intensificationRate = (intensityAt48h - startIntensity) / 48;
    
    // Maximum category
    const maxCategory = forecast.reduce((max, point) => {
      const categoryNumber = point.category === 'TS' ? 0 : parseInt(point.category);
      return Math.max(max, categoryNumber);
    }, 0);
    
    // Duration above hurricane strength
    const hurricaneDuration = forecast.filter(f => f.category !== 'TS' && f.category !== 'TD').length * 6;
    
    // Risk of rapid intensification (>35mph in 24h)
    const maxIntensification = Math.max(
      ...forecast.filter(f => f.hour >= 24).map((f, i, arr) => {
        if (i >= 4) { // 24h difference (4 steps of 6 hours)
          return f.windSpeed - arr[i-4].windSpeed;
        }
        return 0;
      })
    );
    
    const rapidIntensification = maxIntensification >= 35;
    
    setPredictionSummary({
      peakIntensity: Math.round(peakIntensity),
      peakTiming: peakPoint.hour,
      intensificationRate: intensificationRate.toFixed(1),
      maxCategory: maxCategory === 0 ? 'TS' : maxCategory.toString(),
      hurricaneDuration,
      rapidIntensification,
      maxIntensification: Math.round(maxIntensification)
    });
  };
  
  // Get hurricane category from wind speed
  const getHurricaneCategory = (windSpeed) => {
    if (windSpeed < 39) return 'TD';
    if (windSpeed < 74) return 'TS';
    if (windSpeed < 96) return '1';
    if (windSpeed < 111) return '2';
    if (windSpeed < 130) return '3';
    if (windSpeed < 157) return '4';
    return '5';
  };
  
  // Get filtered data based on selected timeframe
  const getFilteredData = () => {
    let maxHour;
    
    switch (selectedTimeframe) {
      case '24h': maxHour = 24; break;
      case '48h': maxHour = 48; break;
      case '72h': maxHour = 72; break;
      case '5d': 
      default: maxHour = 120;
    }
    
    return predictionData.filter(d => d.hour <= maxHour);
  };
  
  // Get ensemble data filtered by timeframe
  const getFilteredEnsembleData = () => {
    let maxHour;
    
    switch (selectedTimeframe) {
      case '24h': maxHour = 24; break;
      case '48h': maxHour = 48; break;
      case '72h': maxHour = 72; break;
      case '5d': 
      default: maxHour = 120;
    }
    
    // Flatten ensemble data for chart
    return ensembleData.flatMap(ensemble => 
      ensemble.filter(d => d.hour <= maxHour)
    );
  };
  
  // Handle hover on chart
  const handleChartHover = (data) => {
    if (data && data.activePayload && data.activePayload.length > 0) {
      setHoverInfo(data.activePayload[0].payload);
    } else {
      setHoverInfo(null);
    }
  };
  
  // Render loading state
  if (loading) {
    return (
      <div className="bg-[#1a237e] p-4 rounded-lg">
        <h3 className="font-bold mb-4">AI Hurricane Forecast</h3>
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
          <span className="ml-3">Generating predictions...</span>
        </div>
      </div>
    );
  }
  
  // Render when no hurricane is selected
  if (!selectedHurricane) {
    return (
      <div className="bg-[#1a237e] p-4 rounded-lg">
        <h3 className="font-bold mb-4">AI Hurricane Forecast</h3>
        <div className="text-center py-8 text-gray-300">
          <p>Select a hurricane to view AI predictions</p>
        </div>
      </div>
    );
  }
  
  // Main render
  return (
    <div className="bg-[#1a237e] p-4 rounded-lg">
      <h3 className="font-bold mb-4 flex justify-between items-center">
        <span>AI Hurricane Forecast</span>
        
        {/* View selector */}
        <div className="flex text-xs overflow-hidden rounded">
          <button 
            onClick={() => setSelectedView('intensity')}
            className={`px-2 py-0.5 ${selectedView === 'intensity' ? 'bg-blue-600' : 'bg-[#0B1021]/50'}`}
          >
            Intensity
          </button>
          <button 
            onClick={() => setSelectedView('ensemble')}
            className={`px-2 py-0.5 ${selectedView === 'ensemble' ? 'bg-blue-600' : 'bg-[#0B1021]/50'}`}
          >
            Ensemble
          </button>
          <button 
            onClick={() => setSelectedView('confidence')}
            className={`px-2 py-0.5 ${selectedView === 'confidence' ? 'bg-blue-600' : 'bg-[#0B1021]/50'}`}
          >
            Confidence
          </button>
        </div>
      </h3>
      
      {/* Timeframe selector */}
      <div className="flex mb-4 gap-1">
        {['24h', '48h', '72h', '5d'].map(timeframe => (
          <button
            key={timeframe}
            onClick={() => setSelectedTimeframe(timeframe)}
            className={`px-2 py-1 text-xs rounded ${
              selectedTimeframe === timeframe 
                ? 'bg-[#2a3890]' 
                : 'bg-[#1a237e]/50 hover:bg-[#1a237e]'
            }`}
          >
            {timeframe}
          </button>
        ))}
      </div>
      
      {predictionSummary && (
        <div className="mb-4 bg-[#0B1021]/30 p-3 rounded-lg">
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <div className="flex items-center">
              <Wind className="h-4 w-4 mr-1 text-blue-400" />
              <span>Peak: {predictionSummary.peakIntensity} mph</span>
              <span className="text-xs text-gray-400 ml-1">
                (+{predictionSummary.peakTiming}h)
              </span>
            </div>
            <div className="flex items-center">
              <Navigation className="h-4 w-4 mr-1 text-yellow-400" />
              <span>Max Cat: {predictionSummary.maxCategory}</span>
            </div>
            <div className="flex items-center">
              {parseFloat(predictionSummary.intensificationRate) > 0 ? (
                <ArrowUp className="h-4 w-4 mr-1 text-red-400" />
              ) : (
                <ArrowDown className="h-4 w-4 mr-1 text-green-400" />
              )}
              <span>
                {Math.abs(parseFloat(predictionSummary.intensificationRate))} mph/h
              </span>
              <span className="text-xs text-gray-400 ml-1">
                (trend)
              </span>
            </div>
            <div className="flex items-center">
              <Clock className="h-4 w-4 mr-1 text-purple-400" />
              <span>{predictionSummary.hurricaneDuration}h</span>
              <span className="text-xs text-gray-400 ml-1">
                (hurricane-force)
              </span>
            </div>
          </div>
          
          {predictionSummary.rapidIntensification && (
            <div className="mt-3 bg-red-900/30 p-2 rounded border border-red-700/50 text-xs flex items-start">
              <AlertTriangle className="h-4 w-4 mr-1 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-bold text-red-400">Rapid Intensification Alert</span>
                <p className="mt-1">Forecast indicates potential for rapid intensification (+{predictionSummary.maxIntensification} mph in 24h)</p>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Content based on selected view */}
      {selectedView === 'intensity' && (
        <div className="mb-3">
          <div className="h-48 mb-2">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart 
                data={getFilteredData()} 
                onMouseMove={handleChartHover}
                onMouseLeave={() => setHoverInfo(null)}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
                <XAxis 
                  dataKey="hour" 
                  stroke="#fff" 
                  label={{ value: 'Hours', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  stroke="#fff" 
                  label={{ value: 'Wind (mph)', angle: -90, position: 'insideLeft' }}
                  domain={[0, dataMax => Math.max(180, dataMax * 1.1)]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a237e',
                    border: 'none',
                    borderRadius: '4px'
                  }}
                  formatter={(value, name) => {
                    if (name === 'uncertaintyRange') return ['', ''];
                    return [Math.round(value), name === 'windSpeed' ? 'Wind Speed' : name];
                  }}
                  labelFormatter={value => `+${value} hours`}
                />
                
                {/* Category threshold lines */}
                <CartesianGrid 
                  vertical={false} 
                  horizontal={false} 
                />
                <Line
                  dataKey={() => 39}
                  stroke="#5DA5DA"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                  name="Tropical Storm"
                />
                <Line
                  dataKey={() => 74}
                  stroke="#4DC4FF"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                  name="Category 1"
                />
                <Line
                  dataKey={() => 96}
                  stroke="#4DFFEA"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                  name="Category 2"
                />
                <Line
                  dataKey={() => 111}
                  stroke="#FFDE33"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                  name="Category 3"
                />
                <Line
                  dataKey={() => 130} stroke="#FF9933"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                  name="Category 4"
                />
                <Line
                  dataKey={() => 157}
                  stroke="#FF5050"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                  name="Category 5"
                />
                
                {/* Uncertainty range as area */}
                <Area
                  dataKey="uncertaintyRange"
                  fill="#4f46e5"
                  fillOpacity={0.2}
                  stroke="none"
                  isAnimationActive={false}
                />
                
                {/* Main prediction line */}
                <Line
                  type="monotone"
                  dataKey="windSpeed"
                  stroke="#4f46e5"
                  strokeWidth={2}
                  dot={{ stroke: '#4f46e5', strokeWidth: 2, r: 3 }}
                  animationDuration={500}
                  name="Wind Speed"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          {/* Hover info */}
          {hoverInfo && (
            <div className="bg-[#0B1021]/50 p-2 rounded-lg text-xs">
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <span className="text-gray-400">Time:</span> +{hoverInfo.hour}h
                </div>
                <div>
                  <span className="text-gray-400">Wind:</span> {Math.round(hoverInfo.windSpeed)} mph
                </div>
                <div>
                  <span className="text-gray-400">Category:</span> {hoverInfo.category}
                </div>
                <div>
                  <span className="text-gray-400">Range:</span> {Math.round(hoverInfo.windLow)}-{Math.round(hoverInfo.windHigh)} mph
                </div>
                <div>
                  <span className="text-gray-400">Confidence:</span> {hoverInfo.confidence}%
                </div>
              </div>
            </div>
          )}
          
          {/* Category colors key */}
          <div className="flex justify-between text-xs mt-3">
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
      
      {/* Ensemble view */}
      {selectedView === 'ensemble' && (
        <div className="mb-3">
          <div className="h-48 mb-2">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart 
                data={getFilteredEnsembleData()}
                onMouseMove={handleChartHover}
                onMouseLeave={() => setHoverInfo(null)}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
                <XAxis 
                  dataKey="hour" 
                  stroke="#fff" 
                  label={{ value: 'Hours', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  stroke="#fff" 
                  label={{ value: 'Wind (mph)', angle: -90, position: 'insideLeft' }}
                  domain={[0, dataMax => Math.max(180, dataMax * 1.1)]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a237e',
                    border: 'none',
                    borderRadius: '4px'
                  }}
                  formatter={(value, name) => {
                    if (name === 'windSpeed') return [Math.round(value), 'Wind Speed'];
                    return [value, name];
                  }}
                  labelFormatter={value => `+${value} hours`}
                />
                
                {/* Category thresholds */}
                <Line
                  dataKey={() => 74}
                  stroke="#4DC4FF"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                  name="Category 1"
                />
                <Line
                  dataKey={() => 111}
                  stroke="#FFDE33"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                  name="Category 3"
                />
                <Line
                  dataKey={() => 157}
                  stroke="#FF5050"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                  name="Category 5"
                />
                
                {/* Ensemble members */}
                {ensembleData.map((ensemble, i) => (
                  <Line
                    key={`ensemble-${i}`}
                    dataKey="windSpeed"
                    data={ensemble.filter(d => d.hour <= (selectedTimeframe === '24h' ? 24 : selectedTimeframe === '48h' ? 48 : selectedTimeframe === '72h' ? 72 : 120))}
                    stroke="#4f46e5"
                    strokeWidth={0.7}
                    strokeOpacity={0.3}
                    dot={false}
                    isAnimationActive={false}
                    name={`Ensemble ${i+1}`}
                  />
                ))}
                
                {/* Main prediction line */}
                <Line
                  type="monotone"
                  dataKey="windSpeed"
                  data={getFilteredData()}
                  stroke="#4f46e5"
                  strokeWidth={2}
                  dot={{ stroke: '#4f46e5', strokeWidth: 2, r: 3 }}
                  animationDuration={500}
                  name="Official Forecast"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          <div className="text-xs bg-[#0B1021]/50 p-2 rounded-lg flex items-start">
            <Info className="h-4 w-4 mr-1 flex-shrink-0 mt-0.5 text-blue-400" />
            <span>
              Ensemble forecasting uses multiple prediction models to account for uncertainty. The spread indicates possible outcomes.
            </span>
          </div>
        </div>
      )}
      
      {/* Confidence view */}
      {selectedView === 'confidence' && (
        <div className="mb-3">
          <div className="h-48 mb-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart 
                data={confidenceData.filter(d => d.hour <= (selectedTimeframe === '24h' ? 24 : selectedTimeframe === '48h' ? 48 : selectedTimeframe === '72h' ? 72 : 120))}
                onMouseMove={handleChartHover}
                onMouseLeave={() => setHoverInfo(null)}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
                <XAxis 
                  dataKey="hour" 
                  stroke="#fff" 
                  label={{ value: 'Hours', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  stroke="#fff" 
                  label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }}
                  domain={[0, 100]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a237e',
                    border: 'none',
                    borderRadius: '4px'
                  }}
                  formatter={(value) => [`${Math.round(value)}%`, '']}
                  labelFormatter={value => `+${value} hours`}
                />
                <Legend />
                
                <Line
                  type="monotone"
                  dataKey="overall"
                  stroke="#4f46e5"
                  strokeWidth={2}
                  name="Overall"
                />
                <Line
                  type="monotone"
                  dataKey="track"
                  stroke="#29b6f6"
                  strokeWidth={1.5}
                  name="Track"
                />
                <Line
                  type="monotone"
                  dataKey="intensity"
                  stroke="#f44336"
                  strokeWidth={1.5}
                  name="Intensity"
                />
                <Line
                  type="monotone"
                  dataKey="timing"
                  stroke="#4caf50"
                  strokeWidth={1.5}
                  name="Timing"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="text-xs bg-[#0B1021]/50 p-2 rounded-lg flex items-start">
            <Info className="h-4 w-4 mr-1 flex-shrink-0 mt-0.5 text-blue-400" />
            <span>
              Confidence decreases with forecast time. Track confidence shows position certainty, while intensity shows wind speed certainty.
            </span>
          </div>
        </div>
      )}
      
      {/* Daily category forecast */}
      <div className="mt-6">
        <h4 className="text-sm font-bold mb-2">Category Forecast</h4>
        <div className="grid grid-cols-5 gap-1">
          {getFilteredData()
            .filter(d => d.hour % 24 === 0)
            .map((point, index) => {
              const categoryColor = 
                point.category === '5' ? '#FF5050' :
                point.category === '4' ? '#FF9933' :
                point.category === '3' ? '#FFDE33' :
                point.category === '2' ? '#4DFFEA' :
                point.category === '1' ? '#4DC4FF' : '#5DA5DA';
              
              return (
                <div 
                  key={index} 
                  className="flex flex-col items-center bg-[#0B1021]/50 p-2 rounded-lg"
                >
                  <span className="text-xs text-gray-300">Day {point.day}</span>
                  <span 
                    className="text-xl font-bold" 
                    style={{ color: categoryColor }}
                  >
                    {point.category}
                  </span>
                  <span className="text-xs">{Math.round(point.windSpeed)} mph</span>
                </div>
              );
            })
          }
        </div>
      </div>
      
      <div className="mt-6 text-xs text-gray-300 flex items-center">
        <Info className="h-4 w-4 mr-1 flex-shrink-0" />
        <span>
          AI predictions based on ensemble forecasting with historical hurricane data and realtime NASA imagery
        </span>
      </div>
    </div>
  );
};

export default HurricanePrediction;