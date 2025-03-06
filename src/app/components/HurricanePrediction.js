'use client';

import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Wind, AlertCircle, ChevronDown, ChevronUp, BarChart4, TrendingUp, Info } from 'lucide-react';
import { HurricaneEnvironment, HurricanePredictionAgent } from '../hurricaneAgent';
import { fetchHistoricalHurricaneData, preprocessDataForTraining } from '../hurricaneData';
import EnsembleVisualization from './EnsembleVisualization';

// Utility function to safely handle potentially NaN values
const safeNumber = (value, fallback = 0) => {
  return typeof value === 'number' && !isNaN(value) ? value : fallback;
};

// Custom transition component that maintains DOM structure during loading
const TransitionWrapper = ({ children, isLoading, className = '' }) => {
  return (
    <div className={`relative transition-opacity duration-300 ${className}`} 
      style={{ opacity: isLoading ? 0 : 1 }}>
      {children}
      {isLoading && (
        <div className="absolute inset-0 bg-[#1a237e] z-10" />
      )}
    </div>
  );
};

const HurricanePrediction = ({ selectedHurricane, nasaService }) => {
  const [agent, setAgent] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [predictions, setPredictions] = useState([]);
  const [ensemblePredictions, setEnsemblePredictions] = useState([]);
  const [forecastStatistics, setForecastStatistics] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [activeTab, setActiveTab] = useState('forecast');
  const [showDetails, setShowDetails] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [dataError, setDataError] = useState(null);
  const [isVisualLoading, setIsVisualLoading] = useState(false);
  
  // Maintain cached content for smoother transitions
  const [cachedContent, setCachedContent] = useState({
    forecast: null,
    ensemble: null,
    performance: null
  });
  
  // Reference to the component mount status
  const isMounted = useRef(false);

  // Initialize agent when component mounts
  useEffect(() => {
    isMounted.current = true;
    
    async function initializeAgent() {
      try {
        if (agent) return; // Only initialize once
        setTrainingStatus('loading');
        setLoadingMessage('Fetching global hurricane data from IBTrACS...');
        
        // Fetch historical data (real data from IBTrACS)
        const historicalData = await fetchHistoricalHurricaneData({
          fullHistory: false,  // Use last 3 years for faster loading
          minTrackPoints: 10   // Ensure sufficient data for training
        });
        
        if (!historicalData || historicalData.length === 0) {
          throw new Error('Failed to load hurricane data');
        }
        
        // Make sure we're still mounted before continuing
        if (!isMounted.current) return;
        
        setLoadingMessage(`Processing ${historicalData.length} historical cyclones...`);
        const processedData = preprocessDataForTraining(historicalData);
        
        // Create environment and agent
        setLoadingMessage('Initializing hurricane environment...');
        const environment = new HurricaneEnvironment();
        await environment.initialize(processedData, nasaService);
        
        const newAgent = new HurricanePredictionAgent({
          useDynamicWeights: true,
          useBasinModels: true,
          ensembleSize: 15
        });
        
        // Train agent
        setTrainingStatus('training');
        setLoadingMessage('Training prediction models...');
        const episodes = 20;
        
        // Set up progress updates
        const progressInterval = setInterval(() => {
          if (!isMounted.current) {
            clearInterval(progressInterval);
            return;
          }
          
          setTrainingProgress(prev => {
            if (prev >= 95) {
              clearInterval(progressInterval);
              return prev;
            }
            return prev + 5;
          });
        }, 400);
        
        try {
          const trainingPerformance = await newAgent.train(environment, episodes);
          
          // Make sure we're still mounted before updating state
          if (!isMounted.current) {
            clearInterval(progressInterval);
            return;
          }
          
          clearInterval(progressInterval);
          setTrainingProgress(100);
          setTrainingStatus('ready');
          setAgent(newAgent);
          
          // Process and display performance metrics
          const periodicPerformance = trainingPerformance.map((p, i) => ({
            episode: i + 1,
            positionError: safeNumber(p.avgPosError),
            intensityError: safeNumber(p.avgIntensityError),
            pressureError: safeNumber(p.avgPressureError, 0)
          }));
          
          // Get performance by forecast period
          const perfByPeriod = trainingPerformance[trainingPerformance.length - 1]?.byForecastPeriod || {};
          
          setPerformance({
            overall: periodicPerformance,
            lastPositionError: safeNumber(trainingPerformance[trainingPerformance.length - 1]?.avgPosError, 0),
            lastIntensityError: safeNumber(trainingPerformance[trainingPerformance.length - 1]?.avgIntensityError, 0),
            byForecastPeriod: perfByPeriod
          });
        } catch (trainingError) {
          console.error('Error during training:', trainingError);
          if (isMounted.current) {
            clearInterval(progressInterval);
            setDataError('Training failed: ' + trainingError.message);
            setTrainingStatus('error');
          }
        }
      } catch (error) {
        console.error('Error initializing prediction agent:', error);
        if (isMounted.current) {
          setDataError('Initialization failed: ' + error.message);
          setTrainingStatus('error');
        }
      }
    }
    
    initializeAgent();
    
    // Cleanup on unmount
    return () => {
      isMounted.current = false;
    };
  }, [agent, nasaService]);

  // Helper function to render forecast content - used for caching
  const renderForecastContent = () => {
    return (
      <>
        {renderForecastTable()}
        
        {/* Basin Information */}
        {selectedHurricane?.coordinates && (
          <div className="mb-4 p-2 bg-[#2a3890]/50 rounded-lg text-xs">
            <p className="flex items-center gap-1">
              <span className="text-gray-300">Basin:</span> 
              <span className="font-bold">{getBasinFromCoordinates(selectedHurricane.coordinates)}</span>
              <span className="text-gray-300 ml-1">
                ({getBasinFullName(getBasinFromCoordinates(selectedHurricane.coordinates))})
              </span>
            </p>
          </div>
        )}
        
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="mt-2 mb-4 text-xs flex items-center gap-1 text-gray-300 hover:text-white"
        >
          {showDetails ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          <span>{showDetails ? 'Hide' : 'Show'} Forecast Details</span>
        </button>
        
        {showDetails && (
          <div className="space-y-1 text-xs text-gray-300 mb-4">
            <p>• Prediction uses historical tropical cyclone tracks from IBTrACS</p>
            <p>• Basin-specific model adjustments for regional patterns</p>
            <p>• Sea surface temperature influence on intensification</p>
            <p>• Ensemble of 15 prediction models for uncertainty estimation</p>
          </div>
        )}
        
        {/* Use EnsembleVisualization component for better visualization */}
        <EnsembleVisualization 
          predictions={predictions}
          ensemblePredictions={ensemblePredictions}
          statistics={forecastStatistics}
        />
      </>
    );
  };

  // Pre-generate content for each tab to avoid reflows during tab switches
  useEffect(() => {
    if (predictions.length > 0) {
      if (!cachedContent.forecast) {
        setCachedContent(prev => ({
          ...prev,
          forecast: renderForecastContent()
        }));
      }
      
      if (!cachedContent.ensemble) {
        setCachedContent(prev => ({
          ...prev,
          ensemble: (
            <EnsembleVisualization 
              predictions={predictions}
              ensemblePredictions={ensemblePredictions}
              statistics={forecastStatistics}
            />
          )
        }));
      }
      
      if (!cachedContent.performance && performance) {
        setCachedContent(prev => ({
          ...prev,
          performance: renderPerformanceMetrics()
        }));
      }
    }
  }, [predictions, ensemblePredictions, forecastStatistics, performance]);

  // Generate predictions when selected hurricane changes
  useEffect(() => {
    if (!agent || !selectedHurricane || trainingStatus !== 'ready') return;
    
    // Don't show loading state immediately to prevent flicker on fast operations
    const loadingTimer = setTimeout(() => {
      if (isMounted.current) {
        setIsVisualLoading(true);
      }
    }, 50);
    
    try {
      // Create prediction for the selected hurricane
      const currentState = {
        position: {
          lat: selectedHurricane.coordinates ? selectedHurricane.coordinates[1] : 0,
          lon: selectedHurricane.coordinates ? selectedHurricane.coordinates[0] : 0
        },
        windSpeed: selectedHurricane.windSpeed || 75,
        pressure: selectedHurricane.pressure || 980,
        basin: getBasinFromCoordinates(selectedHurricane.coordinates),
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
        
        // Calculate uncertainty ranges (increasing with time)
        const uncertaintyFactor = 1 + (day * 0.15);
        const category = getHurricaneCategory(safeNumber(prediction.windSpeed));
        
        newPredictions.push({
          day,
          timestamp,
          position: prediction.position,
          windSpeed: safeNumber(prediction.windSpeed),
          pressure: safeNumber(prediction.pressure),
          category,
          uncertainty: prediction.uncertainty,
          confidence: Math.max(95 - (day * 15), 40)
        });
        
        // Update for next prediction
        history.push({ state: predictState, action: prediction });
        predictState = {
          ...predictState,
          position: prediction.position,
          windSpeed: safeNumber(prediction.windSpeed),
          pressure: safeNumber(prediction.pressure)
        };
      }
      
      setPredictions(newPredictions);
      
      // Generate ensemble predictions
      generateEnsemblePredictions(currentState, forecastDays);
      
      // Clear loading state after content is ready
      clearTimeout(loadingTimer);
      
      // Reset cache to force re-rendering with new data
      setCachedContent({
        forecast: null,
        ensemble: null,
        performance: null
      });
      
      // Allow a little time for DOM updates before removing loading state
      requestAnimationFrame(() => {
        setTimeout(() => {
          if (isMounted.current) {
            setIsVisualLoading(false);
          }
        }, 100);
      });
      
    } catch (error) {
      console.error('Error generating predictions:', error);
      clearTimeout(loadingTimer);
      setIsVisualLoading(false);
    }
    
    return () => clearTimeout(loadingTimer);
  }, [agent, selectedHurricane, trainingStatus]);
  
  // Generate ensemble predictions for visualization
  const generateEnsemblePredictions = (currentState, forecastDays) => {
    if (!agent) return;
    
    try {
      // Create multiple ensemble members
      const ensembleCount = 15;
      const ensembles = [];
      
      for (let i = 0; i < ensembleCount; i++) {
        const ensemblePath = [];
        let ensembleState = {...currentState};
        let ensembleHistory = [];
        
        // Slightly vary the agent parameters for each ensemble member
        const variedAgent = agent.createVariation(0.1 + (Math.random() * 0.2));
        
        for (let day = 1; day <= forecastDays; day++) {
          const prediction = variedAgent.predict(ensembleState, ensembleHistory);
          
          ensemblePath.push({
            day,
            windSpeed: safeNumber(prediction.windSpeed),
            pressure: safeNumber(prediction.pressure),
            position: prediction.position,
            category: getHurricaneCategory(safeNumber(prediction.windSpeed))
          });
          
          ensembleHistory.push({ state: ensembleState, action: prediction });
          ensembleState = {
            ...ensembleState,
            position: prediction.position,
            windSpeed: safeNumber(prediction.windSpeed),
            pressure: safeNumber(prediction.pressure)
          };
        }
        
        ensembles.push(ensemblePath);
      }
      
      setEnsemblePredictions(ensembles);
      
      // Calculate forecast statistics including uncertainty
      if (agent.getForecastStatistics) {
        const statistics = agent.getForecastStatistics(
          ensembles.flat().concat(predictions)
        );
        
        setForecastStatistics(statistics);
      }
      
    } catch (error) {
      console.error('Error generating ensemble predictions:', error);
    }
  };

  // Helper to get hurricane category from wind speed
  const getHurricaneCategory = (windSpeed) => {
    if (windSpeed < 74) return 'TS';
    if (windSpeed < 96) return '1';
    if (windSpeed < 111) return '2';
    if (windSpeed < 130) return '3';
    if (windSpeed < 157) return '4';
    return '5';
  };
  
  // Helper to determine basin from coordinates
  const getBasinFromCoordinates = (coordinates) => {
    if (!coordinates) return 'NA'; // Default to North Atlantic
    
    const [lon, lat] = coordinates;
    
    // Simple basin determination based on coordinates
    if (lon > -100 && lon < 0 && lat > 0) return 'NA'; // North Atlantic
    if (lon >= -180 && lon < -100 && lat > 0) return 'EP'; // Eastern Pacific
    if (lon >= 100 && lon < 180 && lat > 0) return 'WP'; // Western Pacific
    if (lon >= 40 && lon < 100 && lat > 0) return 'NI'; // North Indian
    if (lon >= 40 && lon < 135 && lat <= 0) return 'SI'; // South Indian
    if (lon >= 135 && lon < 180 && lat <= 0) return 'SP'; // South Pacific
    
    // Default to North Atlantic if uncertain
    return 'NA';
  };
  
  // Handle tab switching with smooth transitions
  const switchTab = (tab) => {
    // Don't reload the current tab
    if (tab === activeTab) return;
    
    // Start a brief loading transition
    setIsVisualLoading(true);
    
    // Set the tab after a short delay to allow the fade-out to complete
    setTimeout(() => {
      setActiveTab(tab);
      
      // End the loading state with a short delay to prevent flash
      setTimeout(() => {
        setIsVisualLoading(false);
      }, 50);
    }, 150);
  };

  // Render forecast table
  const renderForecastTable = () => {
    return (
      <div className="mb-4">
        <h4 className="text-sm font-bold mb-2">5-Day Forecast</h4>
        <div className="space-y-2 text-xs">
          {predictions.map(pred => (
            <div key={pred.day} className="flex justify-between items-center">
              <span>Day {pred.day}:</span>
              <div className="flex gap-4">
                <span className="text-gray-300">
                  {safeNumber(pred.position.lat).toFixed(1)}°N, {Math.abs(safeNumber(pred.position.lon)).toFixed(1)}°W
                </span>
                <span>
                  Cat {pred.category} ({Math.round(safeNumber(pred.windSpeed))} mph)
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Render performance metrics
  const renderPerformanceMetrics = () => {
    if (!performance) return null;
    
    return (
      <div className="space-y-4">
        <div>
          <h4 className="text-sm font-bold mb-2 flex items-center gap-1">
            <BarChart4 className="h-4 w-4" />
            <span>Model Performance</span>
          </h4>
          <div className="grid grid-cols-2 gap-2 mb-2 text-xs">
            <div className="bg-[#162040] p-2 rounded-lg">
              <div className="text-gray-400">Position Error</div>
              <div className="text-lg font-bold">{safeNumber(performance.lastPositionError).toFixed(2)} km</div>
            </div>
            <div className="bg-[#162040] p-2 rounded-lg">
              <div className="text-gray-400">Intensity Error</div>
              <div className="text-lg font-bold">{safeNumber(performance.lastIntensityError).toFixed(2)} mph</div>
            </div>
          </div>
          
          <div className="mt-4">
            <h4 className="text-sm font-bold mb-2">Error by Forecast Period</h4>
            <div className="space-y-1 text-xs">
              {Object.entries(performance.byForecastPeriod || {}).map(([period, data]) => (
                data && (
                  <div key={period} className="grid grid-cols-3 bg-[#162040] p-2 rounded-lg">
                    <div>
                      <span className="text-gray-400">Period:</span> {period}
                    </div>
                    <div>
                      <span className="text-gray-400">Position:</span> {safeNumber(data.avgPosError).toFixed(1)} km
                    </div>
                    <div>
                      <span className="text-gray-400">Intensity:</span> {safeNumber(data.avgIntensityError).toFixed(1)} mph
                    </div>
                  </div>
                )
              ))}
            </div>
          </div>
          
          <div className="h-48 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={performance.overall}
                margin={{ top: 5, right: 20, left: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2a4858" />
                <XAxis 
                  dataKey="episode" 
                  label={{ value: 'Training Episode', position: 'insideBottom', offset: -5 }}
                  stroke="#fff" 
                />
                <YAxis 
                  label={{ value: 'Error', angle: -90, position: 'insideLeft' }} 
                  stroke="#fff" 
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a237e',
                    border: 'none',
                    borderRadius: '4px'
                  }}
                  formatter={(value) => safeNumber(value).toFixed(2)}
                />
                <Line
                  type="monotone"
                  dataKey={(dataPoint) => safeNumber(dataPoint.positionError)}
                  name="Position Error (km)"
                  stroke="#29b6f6"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey={(dataPoint) => safeNumber(dataPoint.intensityError)}
                  name="Intensity Error (mph)"
                  stroke="#f44336"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="mt-4 text-xs text-gray-400">
          <p className="font-bold">Model Details:</p>
          <p>• Trained on global IBTrACS historical storm data</p>
          <p>• Ensemble of 15 specialized prediction models</p>
          <p>• Basin-specific parameters for regional accuracy</p>
          <p>• Environmental factors including sea surface temperature</p>
        </div>
      </div>
    );
  };

  // Main render based on status
  if (trainingStatus === 'loading' || trainingStatus === 'training') {
    return (
      <div className="bg-[#1a237e] p-4 rounded-lg">
        <h3 className="font-bold mb-2">Global Hurricane AI Forecast</h3>
        <div className="text-center py-4">
          <p className="mb-2">{loadingMessage}</p>
          <div className="w-full bg-[#0B1021] h-2 mt-2 rounded-full overflow-hidden">
            <div 
              className="bg-blue-500 h-full transition-all duration-300" 
              style={{ width: `${trainingProgress}%` }}
            ></div>
          </div>
          <div className="text-xs text-gray-300 mt-4 flex items-start gap-2">
            <Info className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <div>
              Using IBTrACS global cyclone data from 41 countries and territories 
              for AI model training, spanning all ocean basins and tropical cyclone seasons.
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (trainingStatus === 'error') {
    return (
      <div className="bg-[#1a237e] p-4 rounded-lg">
        <h3 className="font-bold mb-2">Global Hurricane AI Forecast</h3>
        <div className="text-center py-4 text-red-400">
          <p className="mb-2">Error training prediction model.</p>
          {dataError && <p className="text-xs">{dataError}</p>}
          <button 
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-[#2a3890] rounded-lg text-white hover:bg-[#3a4890] transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  // Ready status with predictions
  return (
    <div className="bg-[#1a237e] p-4 rounded-lg">
      <h3 className="font-bold mb-4 flex justify-between items-center">
        <span>Global Hurricane AI Forecast</span>
        <div className="flex gap-2">
          <button
            onClick={() => switchTab('forecast')}
            className={`text-xs px-2 py-1 rounded transition-colors ${
              activeTab === 'forecast' ? 'bg-[#2a3890]' : 'bg-[#0b1021]/40'
            }`}
          >
            Forecast
          </button>
          <button
            onClick={() => switchTab('ensemble')}
            className={`text-xs px-2 py-1 rounded transition-colors ${
              activeTab === 'ensemble' ? 'bg-[#2a3890]' : 'bg-[#0b1021]/40'
            }`}
          >
            Ensemble
          </button>
          <button
            onClick={() => switchTab('performance')}
            className={`text-xs px-2 py-1 rounded transition-colors ${
              activeTab === 'performance' ? 'bg-[#2a3890]' : 'bg-[#0b1021]/40'
            }`}
          >
            Metrics
          </button>
        </div>
      </h3>
      
      {/* Main content with fade transitions */}
      <TransitionWrapper isLoading={isVisualLoading}>
        {predictions.length > 0 ? (
          <div className="min-h-[200px] bg-[#1a237e]">
            {/* Pre-rendered content that is maintained during transitions */}
            {activeTab === 'forecast' && (
              <div className="transition-opacity duration-300">
                {cachedContent.forecast || renderForecastContent()}
              </div>
            )}
            
            {activeTab === 'ensemble' && (
              <div className="transition-opacity duration-300">
                {cachedContent.ensemble || (
                  <EnsembleVisualization 
                    predictions={predictions}
                    ensemblePredictions={ensemblePredictions}
                    statistics={forecastStatistics}
                  />
                )}
              </div>
            )}
            
            {activeTab === 'performance' && (
              <div className="transition-opacity duration-300">
                {cachedContent.performance || renderPerformanceMetrics()}
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-4 bg-[#1a237e]">
            <p>Select a hurricane to view AI forecast</p>
          </div>
        )}
      </TransitionWrapper>
    </div>
  );
};

// Helper function to get basin full name
function getBasinFullName(basinCode) {
  const basinNames = {
    'NA': 'North Atlantic',
    'SA': 'South Atlantic',
    'EP': 'Eastern Pacific',
    'WP': 'Western Pacific',
    'NI': 'North Indian',
    'SI': 'South Indian',
    'SP': 'South Pacific',
    'MM': 'Mediterranean'
  };
  
  return basinNames[basinCode] || basinCode;
}

export default HurricanePrediction;