'use client';

import 'leaflet/dist/leaflet.css';
import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { AlertTriangle } from 'lucide-react';

// Dynamically import map components
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
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

const AtlasCommandMap = ({ hurricanes, selectedHurricane, onSelectHurricane }) => {
  const [animationFrame, setAnimationFrame] = useState(0);
  const [mapReady, setMapReady] = useState(false);
  
  // Animation for pulsing effects
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimationFrame(prev => (prev + 1) % 60);
    }, 50);
    
    return () => clearInterval(interval);
  }, []);
  
  // Notify when map is ready
  useEffect(() => {
    if (typeof window !== 'undefined') {
      setMapReady(true);
    }
  }, []);
  
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

  if (!mapReady) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-[#0d1424]">
        <p>Loading map...</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      <MapContainer
        center={selectedHurricane?.coordinates ? 
          [selectedHurricane.coordinates[1], selectedHurricane.coordinates[0]] : 
          [25, -80]}
        zoom={5}
        className="w-full h-full"
        minZoom={2}
        maxBounds={[[-90, -180], [90, 180]]}
        maxBoundsViscosity={1.0}
        zoomControl={false}
      >
        <ZoomControl position="bottomright" />
        
        {/* Base map layer - dark theme */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          noWrap={true}
        />
        
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
    </div>
  );
};

export default AtlasCommandMap;