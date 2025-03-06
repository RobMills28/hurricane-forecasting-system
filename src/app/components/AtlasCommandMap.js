'use client';

import 'leaflet/dist/leaflet.css';
import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { AlertTriangle, Layers } from 'lucide-react';
import { getGibsLayers, getGibsUrlTemplate } from '../nasaService';

// Dynamically import map components
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { 
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-[#0d1424]">
        <div className="bg-[#0d1424] w-full h-full"></div>
      </div>
    )
  }
);

const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
);

const Circle = dynamic(
  () => import('react-leaflet').then((mod) => mod.Circle),
  { ssr: false }
);

const Popup = dynamic(
  () => import('react-leaflet').then((mod) => mod.Popup),
  { ssr: false }
);

const Polyline = dynamic(
  () => import('react-leaflet').then((mod) => mod.Polyline),
  { ssr: false }
);

const ZoomControl = dynamic(
  () => import('react-leaflet').then((mod) => mod.ZoomControl),
  { ssr: false }
);

// For preloading layers
const preloadImage = (url) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(url);
    img.onerror = () => reject(new Error(`Failed to preload ${url}`));
    img.src = url;
  });
};

const AtlasCommandMap = ({ hurricanes, selectedHurricane, onSelectHurricane }) => {
  const [animationFrame, setAnimationFrame] = useState(0);
  const [mapReady, setMapReady] = useState(false);
  const [nasaLayers, setNasaLayers] = useState([]);
  const [activeBaseMap, setActiveBaseMap] = useState('dark');
  const [activeNasaLayer, setActiveNasaLayer] = useState(null);
  const [showLayerControl, setShowLayerControl] = useState(false);
  
  // Keep both base maps loaded but control opacity
  const [darkMapOpacity, setDarkMapOpacity] = useState(1);
  const [satelliteMapOpacity, setSatelliteMapOpacity] = useState(0);
  const [nasaLayerOpacity, setNasaLayerOpacity] = useState(0);
  const [previousNasaLayer, setPreviousNasaLayer] = useState(null);
  
  // For tile preloading
  const [preloadedUrls, setPreloadedUrls] = useState(new Set());
  
  // Reference to the map container
  const mapRef = useRef(null);
  
  // Animation for pulsing effects
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimationFrame(prev => (prev + 1) % 60);
    }, 50);
    
    return () => clearInterval(interval);
  }, []);
  
  // Pre-load common tile URLs for smoother first appearance
  useEffect(() => {
    const preloadTiles = async () => {
      try {
        // The most common zoom level tiles for initial view
        const darkMapUrl = 'https://a.basemaps.cartocdn.com/dark_all/5/8/12.png';
        const satelliteMapUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/5/12/8';
        
        // Preload these tiles
        await Promise.all([
          preloadImage(darkMapUrl),
          preloadImage(satelliteMapUrl)
        ]);
        
        setPreloadedUrls(prev => new Set([...prev, darkMapUrl, satelliteMapUrl]));
      } catch (error) {
        console.error('Failed to preload tiles:', error);
      }
    };
    
    preloadTiles();
  }, []);
  
  // Fetch NASA GIBS layers
  useEffect(() => {
    const fetchNasaLayers = async () => {
      try {
        const layers = await getGibsLayers();
        setNasaLayers(layers);
      } catch (error) {
        console.error('Error fetching NASA layers:', error);
      }
    };
    
    fetchNasaLayers();
  }, []);
  
  // Handle map's initial ready state with a consistent approach
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Add the main background color to body to prevent any white flashing
      document.body.style.backgroundColor = '#0d1424';
      // Set a short timeout to ensure consistent timing
      const timer = setTimeout(() => {
        setMapReady(true);
      }, 100);
      
      return () => clearTimeout(timer);
    }
  }, []);
  
  // Transition logic for NASA layer changes
  useEffect(() => {
    if (activeNasaLayer) {
      // When a new layer is selected, fade it in
      setNasaLayerOpacity(0.7);
    } else {
      // When layer is removed, fade it out
      setNasaLayerOpacity(0);
    }
    
    // Remember the previous layer for smooth transitions
    if (activeNasaLayer) {
      setPreviousNasaLayer(activeNasaLayer);
    }
  }, [activeNasaLayer]);
  
  // Get hurricane color based on category
  const getHurricaneColor = (hurricane) => {
    const category = hurricane.category || 0;
    
    const colors = {
      0: '#5DA5DA', // Tropical Depression / Tropical Storm (blue)
      1: '#4DC4FF', // Category 1 (lighter blue)
      2: '#4DFFEA', // Category 2 (cyan)
      3: '#FFDE33', // Category 3 (yellow)
      4: '#FF9933', // Category 4 (orange)
      5: '#FF5050'  // Category 5 (red)
    };
    
    return colors[category] || colors[0];
  };
  
  // Generate size based on category for visualization
  const getHurricaneSize = (hurricane) => {
    const baseSize = 80000; // 80km base radius for tropical storm
    const category = hurricane.category || 0;
    const sizeMultiplier = 1 + (category * 0.25); // Size increases with category
    
    // Add subtle pulsing effect based on animation frame
    const pulseEffect = 1 + (Math.sin(animationFrame * 0.05) * 0.1);
    
    return baseSize * sizeMultiplier * (selectedHurricane?.id === hurricane.id ? pulseEffect : 1);
  };
  
  // Generate trajectory prediction (can be enhanced with real forecast data)
  const getPredictedPath = (hurricane) => {
    if (!hurricane.coordinates) return [];
    
    const [lon, lat] = hurricane.coordinates;
    const positions = [[lat, lon]];
    
    // Create a forecast track that curves based on location
    // Northern hemisphere hurricanes tend to curve northeast
    const isNorthernHemisphere = lat > 0;
    
    // Movement parameters
    let latDirection = isNorthernHemisphere ? 1 : -1;
    let lonDirection = isNorthernHemisphere ? 1 : -1;
    
    // For western Atlantic, adjust longitude direction
    if (lon < -30 && lon > -100) {
      lonDirection = isNorthernHemisphere ? 1 : -1;
    }
    
    // Generate forecast points (5 days)
    for (let i = 1; i <= 5; i++) {
      // Add some randomness to make the path look more realistic
      const jitter = (Math.random() - 0.5) * 0.3;
      
      // Curve factor increases with each step
      const curveFactor = i * 0.15;
      
      // Calculate new positions with increasing curve
      const newLat = lat + (i * 0.3 * latDirection) + (curveFactor * latDirection) + jitter;
      const newLon = lon + (i * 0.4 * lonDirection) + jitter;
      
      positions.push([newLat, newLon]);
    }
    
    return positions;
  };

  // Toggle NASA layer selection panel
  const toggleLayerControl = () => {
    setShowLayerControl(!showLayerControl);
  };

  // Set active NASA layer
  const handleLayerSelect = (layer) => {
    // If removing layer
    if (layer === null) {
      setNasaLayerOpacity(0);
      // After fade-out, remove the layer
      setTimeout(() => {
        setActiveNasaLayer(null);
      }, 500);
    } else {
      // If changing layer, set new one first, then fade in
      setActiveNasaLayer(layer);
      requestAnimationFrame(() => {
        setNasaLayerOpacity(0.7);
      });
    }
    setShowLayerControl(false);
  };
  
  // Toggle base map using cross-fade approach
  const changeBaseMap = (mapType) => {
    if (mapType === activeBaseMap) return;
    
    if (mapType === 'dark') {
      // First increase dark map opacity
      setDarkMapOpacity(1);
      // After a short delay, decrease satellite opacity
      setTimeout(() => {
        setSatelliteMapOpacity(0);
      }, 50);
    } else if (mapType === 'satellite') {
      // First increase satellite opacity
      setSatelliteMapOpacity(1);
      // After a short delay, decrease dark map opacity
      setTimeout(() => {
        setDarkMapOpacity(0);
      }, 50);
    }
    setActiveBaseMap(mapType);
  };

  // Render a consistent loading state
  if (!mapReady) {
    return (
      <div className="w-full h-full bg-[#0d1424]">
        <div className="w-full h-full flex items-center justify-center bg-[#0d1424]">
          <p className="text-white">Loading map...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative bg-[#0d1424]">
      <MapContainer
        center={selectedHurricane?.coordinates ? 
          [selectedHurricane.coordinates[1], selectedHurricane.coordinates[0]] : 
          [25, -80]}
        zoom={5}
        className="w-full h-full"
        style={{ background: '#0d1424' }}
        minZoom={2}
        maxBounds={[[-90, -180], [90, 180]]}
        maxBoundsViscosity={1.0}
        zoomControl={false}
        ref={mapRef}
        attributionControl={false}
      >
        <ZoomControl position="bottomright" />
        
        {/* Always include both base maps but control opacity via state */}
        {/* Dark base map */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          noWrap={true}
          opacity={darkMapOpacity}
          className="transition-opacity duration-700"
        />
        
        {/* Satellite base map */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution='Imagery &copy; Esri'
          noWrap={true}
          opacity={satelliteMapOpacity}
          className="transition-opacity duration-700"
        />
        
        {/* NASA Layer - maintained with opacity transitions */}
        {(activeNasaLayer || previousNasaLayer) && (
          <TileLayer
            url={getGibsUrlTemplate(activeNasaLayer || previousNasaLayer)}
            attribution={`NASA GIBS - ${(activeNasaLayer || previousNasaLayer).subtitle}`}
            opacity={nasaLayerOpacity}
            noWrap={true}
            className="transition-opacity duration-700" 
          />
        )}
        
        {/* Hurricanes */}
        {hurricanes.map(hurricane => (
          hurricane.coordinates && (
            <React.Fragment key={hurricane.id}>
              {/* Main hurricane visualization */}
              <Circle
                center={[hurricane.coordinates[1], hurricane.coordinates[0]]}
                radius={getHurricaneSize(hurricane)}
                pathOptions={{
                  color: getHurricaneColor(hurricane),
                  fillColor: getHurricaneColor(hurricane),
                  fillOpacity: 0.3,
                  weight: 2
                }}
                eventHandlers={{
                  click: () => onSelectHurricane(hurricane)
                }}
              >
                <Popup className="custom-popup">
                  <div className="bg-[#1a237e] text-white p-4 rounded-lg shadow-lg min-w-[200px]">
                    <h3 className="font-bold text-lg mb-2 flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 text-yellow-400" />
                      {hurricane.name}
                    </h3>
                    <div className="space-y-2">
                      <p><span className="text-gray-400">Type:</span> {hurricane.type}</p>
                      <p><span className="text-gray-400">Category:</span> {hurricane.category || 'TS'}</p>
                      <p><span className="text-gray-400">Status:</span> {hurricane.status || 'Active'}</p>
                      <p><span className="text-gray-400">Area:</span> {hurricane.areas}</p>
                    </div>
                  </div>
                </Popup>
              </Circle>
              
              {/* Hurricane eye */}
              <Circle
                center={[hurricane.coordinates[1], hurricane.coordinates[0]]}
                radius={10000} // 10km for the eye
                pathOptions={{
                  color: '#ffffff',
                  fillColor: '#ffffff',
                  fillOpacity: 0.7,
                  weight: 1
                }}
              />
              
              {/* Forecast track for selected hurricane */}
              {selectedHurricane?.id === hurricane.id && (
                <Polyline
                  positions={getPredictedPath(hurricane)}
                  pathOptions={{
                    color: '#ffffff',
                    weight: 2.5,
                    opacity: 0.7,
                    dashArray: '10, 10'
                  }}
                />
              )}
              
              {/* Forecast points */}
              {selectedHurricane?.id === hurricane.id && 
                getPredictedPath(hurricane).map((point, index) => {
                  if (index === 0) return null; // Skip the current position
                  
                  return (
                    <Circle
                      key={`forecast-${hurricane.id}-${index}`}
                      center={point}
                      radius={10000}
                      pathOptions={{
                        color: '#ffffff',
                        fillColor: getHurricaneColor(hurricane),
                        fillOpacity: 0.8,
                        weight: 1.5
                      }}
                    >
                      <Popup className="custom-popup">
                        <div className="bg-[#1a237e] text-white p-2 rounded shadow">
                          <p className="font-bold">{hurricane.name}</p>
                          <p>Forecast: +{index * 24}h</p>
                        </div>
                      </Popup>
                    </Circle>
                  );
                })
              }
            </React.Fragment>
          )
        ))}
      </MapContainer>
      
      {/* Layer Controls Panel */}
      <div className="absolute top-2 right-2 bg-[#1a237e] p-2 rounded-lg z-[1000] flex flex-col gap-2">
        {/* Base Map Selection */}
        <div className="flex gap-1">
          <button 
            onClick={() => changeBaseMap('dark')}
            className={`px-2 py-1 rounded text-xs transition-colors ${
              activeBaseMap === 'dark' ? 'bg-[#2a3890] text-white' : 'bg-[#0b1021] text-gray-300'
            }`}
          >
            Dark Map
          </button>
          <button 
            onClick={() => changeBaseMap('satellite')}
            className={`px-2 py-1 rounded text-xs transition-colors ${
              activeBaseMap === 'satellite' ? 'bg-[#2a3890] text-white' : 'bg-[#0b1021] text-gray-300'
            }`}
          >
            Satellite
          </button>
        </div>
        
        {/* NASA Layer Button */}
        <button 
          onClick={toggleLayerControl}
          className="flex items-center gap-1 px-2 py-1 rounded bg-[#0b1021] text-white hover:bg-[#2a3890] transition-colors text-xs"
        >
          <Layers className="h-4 w-4" />
          <span>NASA Layers</span>
        </button>
      </div>
      
      {/* NASA Layer Selection Panel */}
      {showLayerControl && (
        <div className="absolute top-24 right-2 bg-[#0b1021]/90 p-3 rounded-lg z-[1000] max-w-[250px] max-h-[400px] overflow-y-auto">
          <h4 className="text-sm font-bold mb-2 text-white">NASA Earth Observation</h4>
          <div className="space-y-2">
            {nasaLayers.map(layer => (
              <button
                key={layer.id}
                onClick={() => handleLayerSelect(layer)}
                className={`w-full text-left p-2 rounded text-xs transition-colors ${
                  activeNasaLayer?.id === layer.id 
                    ? 'bg-[#1a237e] text-white' 
                    : 'bg-[#162040] text-gray-300 hover:bg-[#1a237e]/50'
                }`}
              >
                <div className="font-bold">{layer.title}</div>
                <div className="text-xs opacity-70">{layer.description}</div>
              </button>
            ))}
            
            {/* Clear Layer Option */}
            {activeNasaLayer && (
              <button
                onClick={() => handleLayerSelect(null)}
                className="w-full text-left p-2 rounded text-xs bg-[#3e1a1a] text-gray-300 hover:bg-[#641f1f] transition-colors"
              >
                <div className="font-bold">Clear NASA Layer</div>
              </button>
            )}
          </div>
        </div>
      )}
      
      {/* Legend */}
      <div className="absolute bottom-2 right-2 bg-[#0b1021]/70 text-white p-3 rounded-lg z-[1000]">
        <h4 className="text-xs font-bold mb-2">Hurricane Categories</h4>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#5DA5DA' }}></div>
            <span>Tropical Storm</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#4DC4FF' }}></div>
            <span>Category 1</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#4DFFEA' }}></div>
            <span>Category 2</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#FFDE33' }}></div>
            <span>Category 3</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#FF9933' }}></div>
            <span>Category 4</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#FF5050' }}></div>
            <span>Category 5</span>
          </div>
        </div>
      </div>
      
      {/* NASA Data Credit */}
      {activeNasaLayer && (
        <div className="absolute bottom-36 right-2 bg-[#0b1021]/70 text-white p-2 rounded-lg z-[1000]">
          <h4 className="text-xs font-bold mb-1">Active NASA Layer</h4>
          <div className="text-xs">
            <p>{activeNasaLayer.title}</p>
            <p className="text-xs opacity-70">{activeNasaLayer.subtitle}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AtlasCommandMap;