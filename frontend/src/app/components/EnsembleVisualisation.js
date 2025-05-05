'use client';

import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, Area, ComposedChart, ReferenceLine, Label, Legend 
} from 'recharts';
import { Wind, AlertCircle, ChevronDown, ChevronUp, BarChart4, TrendingUp } from 'lucide-react';

// Utility function to safely handle potentially NaN values
const safeNumber = (value, fallback = 0) => {
  return typeof value === 'number' && !isNaN(value) ? value : fallback;
};

const EnsembleVisualization = ({ predictions, ensemblePredictions, statistics }) => {
  const [showEnsemble, setShowEnsemble] = useState(true);
  const [showUncertainty, setShowUncertainty] = useState(true);
  
  // Safe value checking
  const safePredictions = Array.isArray(predictions) ? predictions : [];
  const safeEnsemblePredictions = Array.isArray(ensemblePredictions) ? ensemblePredictions : [];
  const safeStatistics = statistics || {};
  
  // Format data for the combined intensity chart
  const prepareIntensityData = () => {
    if (!safePredictions.length) return [];
    
    return safePredictions.map(pred => {
      const day = safeNumber(pred.day, 0);
      const stats = safeStatistics[day] || {};
      const windSpeed = safeNumber(pred.windSpeed, 0);
      
      // Make sure we have valid numbers for all values
      const low = safeNumber(stats.intensity?.range?.[0], windSpeed * 0.8);
      const high = safeNumber(stats.intensity?.range?.[1], windSpeed * 1.2);
      const confidence = safeNumber(stats.confidence, 50);
      
      return {
        day: day,
        windSpeed: windSpeed,
        low: low,
        high: high,
        confidence: confidence,
        category: pred.category || 'TS'
      };
    });
  };
  
  // Format ensemble data for visualization
  const prepareEnsembleData = () => {
    if (!safeEnsemblePredictions.length) return [];
    
    // Flatten ensemble tracks for visualization
    const flatData = [];
    
    safeEnsemblePredictions.forEach((ensemble, ensembleIndex) => {
      if (!Array.isArray(ensemble)) return;
      
      ensemble.forEach(point => {
        // Validate point data
        if (!point || typeof point !== 'object') return;
        
        flatData.push({
          ...point,
          day: safeNumber(point.day, 0),
          windSpeed: safeNumber(point.windSpeed, 0),
          ensembleId: ensembleIndex
        });
      });
    });
    
    return flatData;
  };
  
  // Format data for hurricane categories
  const prepareCategoryData = () => {
    if (!safePredictions.length) return [];
    
    const categories = {
      'TS': { min: 39, max: 73, color: '#5DA5DA', label: 'Tropical Storm' },
      '1': { min: 74, max: 95, color: '#4DC4FF', label: 'Category 1' },
      '2': { min: 96, max: 110, color: '#4DFFEA', label: 'Category 2' },
      '3': { min: 111, max: 129, color: '#FFDE33', label: 'Category 3' },
      '4': { min: 130, max: 156, color: '#FF9933', label: 'Category 4' },
      '5': { min: 157, max: 200, color: '#FF5050', label: 'Category 5' }
    };
    
    return safePredictions.map(pred => {
      const day = safeNumber(pred.day, 0);
      const categoryStat = safeStatistics[day]?.category || {};
      const catRange = categoryStat.range || []; 
      const minCat = typeof catRange[0] === 'string' ? catRange[0] : String(catRange[0] || '0');
      const maxCat = typeof catRange[1] === 'string' ? catRange[1] : String(catRange[1] || '0');
      const category = pred.category || 'TS';
      
      // Define category colors
      const catInfo = categories[category] || categories['TS'];
        
      return {
        day,
        windSpeed: safeNumber(pred.windSpeed, 0),
        category,
        minCategory: minCat,
        maxCategory: maxCat,
        categoryColor: catInfo.color,
        categoryLabel: catInfo.label
      };
    });
  };
  
  // Prepare confidence visualization
  const prepareConfidenceData = () => {
    if (!safeStatistics || typeof safeStatistics !== 'object') return [];
    
    return Object.entries(safeStatistics).map(([day, stat]) => {
      if (!stat || typeof stat !== 'object') {
        return {
          day: parseInt(day) || 0,
          confidence: 50,
          positionConfidence: 50,
          intensityConfidence: 50
        };
      }
      
      return {
        day: parseInt(day) || 0,
        confidence: safeNumber(stat.confidence, 50),
        positionConfidence: safeNumber(stat.position?.confidence, 50),
        intensityConfidence: safeNumber(stat.intensity?.confidence, 50)
      };
    });
  };
  
  const intensityData = prepareIntensityData();
  const ensembleData = prepareEnsembleData();
  const categoryData = prepareCategoryData();
  const confidenceData = prepareConfidenceData();
  
  // Custom tooltip for the intensity chart
  const IntensityTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      return (
        <div className="bg-[#1a237e] text-white p-3 rounded-lg shadow-lg text-xs">
          <p className="font-bold">Day {label}</p>
          <p>Wind Speed: {Math.round(safeNumber(data.windSpeed, 0))} mph</p>
          <p>Range: {Math.round(safeNumber(data.low, 0))} - {Math.round(safeNumber(data.high, 0))} mph</p>
          <p>Confidence: {Math.round(safeNumber(data.confidence, 0))}%</p>
          <p>Category: {data.category || 'TS'}</p>
        </div>
      );
    }
  
    return null;
  };
  
  if (!safePredictions.length) {
    return (
      <div className="text-sm text-center text-gray-400 py-4">
        No forecast data available
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      {/* Intensity Forecast Chart with Uncertainty */}
      <div>
        <div className="flex justify-between items-center mb-2">
          <h4 className="text-sm font-bold">Intensity Forecast</h4>
          <div className="flex gap-2">
            <button
              onClick={() => setShowEnsemble(!showEnsemble)}
              className={`text-xs px-2 py-1 rounded ${
                showEnsemble ? 'bg-[#2a3890]' : 'bg-[#0b1021]/40'
              }`}
            >
              Ensemble
            </button>
            <button
              onClick={() => setShowUncertainty(!showUncertainty)}
              className={`text-xs px-2 py-1 rounded ${
                showUncertainty ? 'bg-[#2a3890]' : 'bg-[#0b1021]/40'
              }`}
            >
              Uncertainty
            </button>
          </div>
        </div>
        
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={intensityData} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
              <XAxis 
                dataKey="day" 
                domain={[1, 5]} 
                ticks={[1, 2, 3, 4, 5]}
                label={{ value: 'Forecast Day', position: 'insideBottom', offset: -5 }}
                stroke="#fff" 
              />
              <YAxis 
                label={{ value: 'Wind Speed (mph)', angle: -90, position: 'insideLeft' }} 
                stroke="#fff" 
                domain={[0, 'dataMax + 20']}
              />
              <Tooltip content={<IntensityTooltip />} />
              <Legend />
              
              {/* Category threshold lines */}
              <ReferenceLine y={74} stroke="#4DC4FF" strokeDasharray="3 3">
                <Label value="Cat 1" position="right" fill="#4DC4FF" />
              </ReferenceLine>
              <ReferenceLine y={96} stroke="#4DFFEA" strokeDasharray="3 3">
                <Label value="Cat 2" position="right" fill="#4DFFEA" />
              </ReferenceLine>
              <ReferenceLine y={111} stroke="#FFDE33" strokeDasharray="3 3">
                <Label value="Cat 3" position="right" fill="#FFDE33" />
              </ReferenceLine>
              <ReferenceLine y={130} stroke="#FF9933" strokeDasharray="3 3">
                <Label value="Cat 4" position="right" fill="#FF9933" />
              </ReferenceLine>
              <ReferenceLine y={157} stroke="#FF5050" strokeDasharray="3 3">
                <Label value="Cat 5" position="right" fill="#FF5050" />
              </ReferenceLine>
              
              {/* Uncertainty area */}
              {showUncertainty && (
                <>
                  <Area 
                    type="monotone" 
                    dataKey="high" 
                    stroke="none"
                    fill="#4f46e5" 
                    fillOpacity={0.2}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="low" 
                    stroke="none"
                    fill="#4f46e5" 
                    fillOpacity={0.2}
                  />
                </>
              )}
              
              {/* Ensemble lines */}
              {showEnsemble && safeEnsemblePredictions.length > 0 && 
                Array.from({ length: Math.min(safeEnsemblePredictions.length, 15) }).map((_, index) => (
                  <Line
                    key={`ensemble-${index}`}
                    data={ensembleData.filter(d => d.ensembleId === index)}
                    type="monotone"
                    dataKey="windSpeed"
                    stroke="#4f46e5"
                    strokeWidth={1}
                    strokeOpacity={0.2}
                    dot={false}
                    activeDot={false}
                    name={`Ensemble ${index + 1}`}
                    hide={!showEnsemble}
                  />
                ))
              }
              
              {/* Main prediction line */}
              <Line
                type="monotone"
                dataKey="windSpeed"
                stroke="#4f46e5"
                strokeWidth={3}
                dot={{ stroke: '#4f46e5', strokeWidth: 2, r: 4 }}
                name="Official Forecast"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Hurricane Categories Visualization */}
      <div className="mt-4">
        <h4 className="text-sm font-bold mb-2">Hurricane Category Forecast</h4>
        <div className="grid grid-cols-5 gap-1 mb-2">
          {categoryData.map(cat => (
            <div 
              key={`cat-${cat.day}`} 
              className="flex flex-col items-center p-2 rounded-lg"
              style={{ backgroundColor: cat.categoryColor + '33' }} // Add transparency
            >
              <div className="text-xs text-gray-300">Day {cat.day}</div>
              <div className="text-xl font-bold" style={{ color: cat.categoryColor }}>
                {cat.category}
              </div>
              <div className="text-xs">
                {cat.minCategory !== cat.maxCategory 
                  ? `${cat.minCategory}-${cat.maxCategory}` 
                  : cat.category
                }
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Forecast Confidence Visualization */}
      {confidenceData.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-bold mb-2">Forecast Confidence</h4>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={confidenceData} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
                <XAxis 
                  dataKey="day" 
                  domain={[1, 5]} 
                  ticks={[1, 2, 3, 4, 5]}
                  stroke="#fff" 
                />
                <YAxis 
                  label={{ value: 'Confidence %', angle: -90, position: 'insideLeft' }} 
                  domain={[0, 100]}
                  stroke="#fff" 
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#1a237e',
                    border: 'none',
                    borderRadius: '4px'
                  }}
                />
                <Legend />
                
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="#00e676"
                  strokeWidth={2}
                  name="Overall"
                />
                <Line
                  type="monotone"
                  dataKey="positionConfidence"
                  stroke="#29b6f6"
                  strokeWidth={2}
                  name="Position"
                />
                <Line
                  type="monotone"
                  dataKey="intensityConfidence"
                  stroke="#f44336"
                  strokeWidth={2}
                  name="Intensity"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnsembleVisualization;