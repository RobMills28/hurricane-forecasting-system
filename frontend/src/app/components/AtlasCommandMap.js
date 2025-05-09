'use client';

import 'leaflet/dist/leaflet.css';
import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { AlertTriangle, Layers, Search, Clock, Filter, Info } from 'lucide-react';
import { getGibsLayers, getGibsUrlTemplate } from '../nasaService';
import { useMap } from 'react-leaflet';

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

const CircleMarker = dynamic(
  () => import('react-leaflet').then((mod) => mod.CircleMarker),
  { ssr: false }
);

const Popup = dynamic(
  () => import('react-leaflet').then((mod) => mod.Popup),
  { ssr: false }
);

const Tooltip = dynamic(
  () => import('react-leaflet').then((mod) => mod.Tooltip),
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

  const AtlasCommandMap = ({ hurricanes, selectedHurricane, onSelectHurricane, potentialStormAreas = [] }) => {
  const [animationFrame, setAnimationFrame] = useState(0);
  const [mapReady, setMapReady] = useState(false);
  const [nasaLayers, setNasaLayers] = useState([]);
  const [activeBaseMap, setActiveBaseMap] = useState('dark');
  const [activeNasaLayer, setActiveNasaLayer] = useState(null);
  const [showLayerControl, setShowLayerControl] = useState(false);
  const [showForecastPaths, setShowForecastPaths] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filterCategory, setFilterCategory] = useState(null);
  const [filterRegion, setFilterRegion] = useState(null);
  const [filterStormType, setFilterStormType] = useState(null); // New filter for storm type
  const [showTimeline, setShowTimeline] = useState(false);
  const [currentTimelineValue, setCurrentTimelineValue] = useState(50); // Middle position (today)
  const [timelineMode, setTimelineMode] = useState('present'); // '-48h', '-24h', 'present', '+24h', '+48h', etc.
  
  // Keep both base maps loaded but control opacity
  const [darkMapOpacity, setDarkMapOpacity] = useState(1);
  const [satelliteMapOpacity, setSatelliteMapOpacity] = useState(0);
  const [nasaLayerOpacity, setNasaLayerOpacity] = useState(0);
  const [previousNasaLayer, setPreviousNasaLayer] = useState(null);
  
  // For tile preloading
  const [preloadedUrls, setPreloadedUrls] = useState(new Set());
  
  // Reference to the map container
  const mapRef = useRef(null);
  
  // Component to handle map clicks for deselection
  const MapClickHandler = () => {
    const map = useMap();
    
    useEffect(() => {
      if (!map) return;
      
      const handleMapClick = (e) => {
        // Don't deselect if clicking on a marker or popup
        if (e.originalEvent.target.classList.contains('leaflet-marker-icon') ||
            e.originalEvent.target.closest('.leaflet-popup')) {
          return;
        }
        // Deselect the hurricane
        if (selectedHurricane) {
          onSelectHurricane(null);
        }
      };
      
      map.on('click', handleMapClick);
      
      return () => {
        map.off('click', handleMapClick);
      };
    }, [map, selectedHurricane]);
    
    return null;
  };
  
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
  
  // Filter hurricanes based on search and filters
  const getFilteredHurricanes = () => {
    if (!hurricanes) return [];
    
    return hurricanes.filter(hurricane => {
      // Search by name
      if (searchQuery && !hurricane.name?.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }
      
      // Filter by category
      if (filterCategory !== null && hurricane.category !== filterCategory) {
        return false;
      }
      
      // Filter by region/basin
      if (filterRegion && hurricane.basin !== filterRegion) {
        return false;
      }
      
      // Filter by storm type
      if (filterStormType && hurricane.type !== filterStormType) {
        return false;
      }
      
      return true;
    });
  };

  const getTimelineFilteredHurricanes = () => {
    const filtered = getFilteredHurricanes();
    
    if (!filtered) return [];
    
    // Here I am simulating time-based filtering
    // If I have an opportunity I should compare actual timestamps
    
    // Specific filtering based on timeline mode
    switch (timelineMode) {
      case '-72h':
        // Show storms from 48-72 hours ago
        // Include global storms with equal distribution
        return filtered.filter((h, index) => 
          // Keep 40% of storms with focus on global distribution
          index % 5 < 2 || h.basin !== 'NA'
        );
      
      case '-48h':
        // Show storms from 24-48 hours ago
        return filtered.filter((h, index) => 
          // Keep 60% of storms with focus on global distribution
          index % 5 < 3 || h.basin !== 'NA'
        );
        
      case '-24h':
        // Show storms from 0-24 hours ago
        return filtered.filter((h, index) => 
          // Keep 80% of storms
          index % 5 < 4
        );
      
      case 'present':
        // Show all current active storms
        return filtered;
      
      case '+24h':
        // Show 24-hour forecast storms
        // Focus on stronger storms and ensure global representation
        return filtered.filter(h => 
          h.category > 0 || 
          h.category === 'TS' || 
          h.basin !== 'NA' // Ensure the system is getting non-US storms
        );
      
      case '+48h':
        // Show 48-hour forecast storms
        // More strict filtering, but ensure global representation
        return filtered.filter(h => 
          (h.category > 1 || h.category === 'TS') || 
          (h.basin !== 'NA' && (h.category === 'TS' || h.category > 0))
        );
        
      case '+72h':
        // Show 72-hour forecast storms
        // Only keep stronger storms, but ensure global representation
        return filtered.filter(h => 
          h.category > 2 || 
          (h.basin !== 'NA' && h.category > 0) // Keep stronger non-US storms
        );
      
      default:
        return filtered;
    }
  };

  const getVisiblePotentialStormAreas = () => {
    // Only show potential areas in future modes
    if (!timelineMode.startsWith('+')) {
      return [];
    }
    
    // Parse the hour value from timelineMode
    const hourValue = parseInt(timelineMode.replace('+', ''));
    
    // Filter areas based on probability threshold that changes with time
    // Further in the future = show more areas with lower probability
    let probabilityThreshold;
    
    if (hourValue <= 24) {
      probabilityThreshold = 0.6; // Only high probability areas for near future
    } else if (hourValue <= 48) {
      probabilityThreshold = 0.4; // Medium probability for mid-future
    } else {
      probabilityThreshold = 0.2; // Lower probability for far future
    }
    
    return potentialStormAreas.filter(area => area.probability >= probabilityThreshold);
  };

  const getPulseAnimationClass = (probability) => {
    // Animation is stronger for higher probability areas
    const intensityClass = probability > 0.6 ? 'bg-red-500' : 
                          probability > 0.4 ? 'bg-orange-500' : 'bg-yellow-500';
    
    // Use animation frame to create pulse effect
    const frame = animationFrame % 60;
    
    if (frame < 15) {
      return `${intensityClass} opacity-30 scale-75`;
    } else if (frame < 30) {
      return `${intensityClass} opacity-60 scale-100`;
    } else if (frame < 45) {
      return `${intensityClass} opacity-90 scale-125`;
    } else {
      return `${intensityClass} opacity-60 scale-100`;
    }
  };  
    
  // Get hurricane colour based on category and storm type
  const getHurricaneColor = (hurricane) => {
    // Check for storm type first
    if (hurricane.type === 'Winter Storm') {
      return '#87CEFA'; // Light blue for winter storms
    }
    
    if (hurricane.type === 'Severe Thunderstorm') {
      return '#9932CC'; // Purple for severe thunderstorms
    }
    
    if (hurricane.type === 'Severe Storm') {
      return '#6495ED'; // Cornflower blue for severe storms
    }
    
    // For tropical systems, colour by category
    const category = hurricane.category || 0;
    
    const colors = {
      'TD': '#5DA5DA', // Tropical Depression (blue)
      'TS': '#5DA5DA', // Tropical Storm (blue)
      0: '#5DA5DA',   // For numerical fallback
      1: '#4DC4FF',   // Category 1 (lighter blue)
      2: '#4DFFEA',   // Category 2 (cyan)
      3: '#FFDE33',   // Category 3 (yellow)
      4: '#FF9933',   // Category 4 (orange)
      5: '#FF5050'    // Category 5 (red)
    };
    
    return colors[category] || colors['TS'];
  };
  
  // Get marker size based on category and selection
  const getMarkerSize = (hurricane) => {
    // Base size is slightly larger for higher categories
    let baseSize;
    
    // Different size based on storm type
    if (hurricane.type === 'Winter Storm' || hurricane.type === 'Severe Storm') {
      baseSize = 5; // Standard size for non-tropical systems
    } else {
      // For tropical systems, scale by category
      const categoryValue = hurricane.category === 'TS' ? 0 : 
                           hurricane.category === 'TD' ? 0 : 
                           parseInt(hurricane.category) || 0;
      baseSize = 5 + (categoryValue * 0.8);
    }
    
    // Make selected hurricane slightly larger
    if (selectedHurricane?.id === hurricane.id) {
      return baseSize + 2;
    }
    
    return baseSize;
  };
  
  // Get marker opacity based on selection
  const getMarkerOpacity = (hurricane) => {
    // Make selected hurricane fully opaque
    if (selectedHurricane?.id === hurricane.id) {
      return 1;
    }
    
    // If a hurricane is selected, make others more transparent
    if (selectedHurricane && selectedHurricane.id !== hurricane.id) {
      return 0.5;
    }
    
    // Default opacity varies slightly based on category for visual hierarchy
    const categoryValue = hurricane.category === 'TS' ? 0 : 
                         hurricane.category === 'TD' ? 0 : 
                         parseInt(hurricane.category) || 0;
    return 0.7 + (categoryValue * 0.05);
  };
  
  // Get stroke width based on selection
  const getStrokeWidth = (hurricane) => {
    return selectedHurricane?.id === hurricane.id ? 2 : 1;
  };

  // Generate trajectory prediction
  const getPredictedPath = (hurricane) => {
    if (!hurricane.coordinates) return [];
    
    const [lon, lat] = hurricane.coordinates;
    const positions = [[lat, lon]];
    
    // Create a forecast track that curves based on location
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
      // Here I am adding some randomness to make the path look more realistic
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
    
    // Close other panels when opening this one
    if (!showLayerControl) {
      setShowSearch(false);
      setShowFilters(false);
      setShowTimeline(false);
    }
  };
  
  // Toggle search panel
  const toggleSearch = () => {
    setShowSearch(!showSearch);
    
    // Close other panels when opening this one
    if (!showSearch) {
      setShowLayerControl(false);
      setShowFilters(false);
      setShowTimeline(false);
    }
  };
  
  // Toggle filters panel
  const toggleFilters = () => {
    setShowFilters(!showFilters);
    
    // Close other panels when opening this one
    if (!showFilters) {
      setShowLayerControl(false);
      setShowSearch(false);
      setShowTimeline(false);
    }
  };
  
  // Toggle timeline
  const toggleTimeline = () => {
    setShowTimeline(!showTimeline);
    
    // Close other panels when opening this one
    if (!showTimeline) {
      setShowLayerControl(false);
      setShowSearch(false);
      setShowFilters(false);
    }
  };

  const handleTimelineChange = (e) => {
    const newValue = parseInt(e.target.value);
    setCurrentTimelineValue(newValue);
    
    // More granular time periods
    if (newValue < 15) {
      setTimelineMode('-72h'); // 72 hours in the past
    } else if (newValue < 30) {
      setTimelineMode('-48h'); // 48 hours in the past
    } else if (newValue < 45) {
      setTimelineMode('-24h'); // 24 hours in the past
    } else if (newValue <= 55) {
      setTimelineMode('present'); // Current
    } else if (newValue < 70) {
      setTimelineMode('+24h'); // 24 hours in the future
    } else if (newValue < 85) {
      setTimelineMode('+48h'); // 48 hours in the future
    } else {
      setTimelineMode('+72h'); // 72 hours in the future
    }
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
  
  // Toggle forecast paths view
  const toggleForecastPaths = () => {
    setShowForecastPaths(!showForecastPaths);
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

  // Get filtered hurricanes
  const filteredHurricanes = getFilteredHurricanes();

  // Get unique storm types for filter
  const stormTypes = Array.from(
    new Set(
      hurricanes
        ?.filter(h => h.type)
        .map(h => h.type) || []
    )
  ).sort();

  return (
    <div className="w-full h-full relative bg-[#0d1424]">
      <MapContainer
        center={selectedHurricane?.coordinates ? 
          [selectedHurricane.coordinates[1], selectedHurricane.coordinates[0]] : 
          [25, -40]} // Shifted initial view to show more of the Atlantic and global view
        zoom={3} // Decreased initial zoom to show more of the world
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
        <MapClickHandler />
        
        {/* Base maps with opacity control */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          noWrap={true}
          opacity={darkMapOpacity}
          className="transition-opacity duration-700"
        />
        
        <TileLayer
          url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
          attribution='Imagery &copy; Esri'
          noWrap={true}
          opacity={satelliteMapOpacity}
          className="transition-opacity duration-700"
        />
        
        {/* NASA Layer with opacity transitions */}
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
  {getTimelineFilteredHurricanes().map(hurricane => (
    hurricane.coordinates && (
      <React.Fragment key={`marker-${hurricane.id}`}>
        {/* Main hurricane marker - Always using CircleMarker for consistent display */}
        <CircleMarker
          center={[hurricane.coordinates[1], hurricane.coordinates[0]]}
          radius={getMarkerSize(hurricane) + 2} // Slightly larger dots
          pathOptions={{
            color: '#ffffff',
            weight: getStrokeWidth(hurricane),
            fillColor: getHurricaneColor(hurricane),
            fillOpacity: getMarkerOpacity(hurricane)
          }}
          eventHandlers={{
            click: () => onSelectHurricane(hurricane)
          }}
          className="cursor-pointer"
        >
          {/* Add an invisible larger hit area */}
          <CircleMarker
            center={[hurricane.coordinates[1], hurricane.coordinates[0]]}
            radius={15} // Large invisible hit area (15px radius)
            pathOptions={{
              color: 'transparent',
              fillColor: 'transparent',
              fillOpacity: 0
            }}
            eventHandlers={{
              click: () => onSelectHurricane(hurricane)
            }}
          />
            {/* Tooltip on hover */}
            <Tooltip 
              direction="top" 
              offset={[0, -5]} 
              opacity={0.9} 
              className="custom-tooltip"
            >
              <div className="px-2 py-1">
                <p className="font-bold">{hurricane.name}</p>
                <p className="text-sm">Type: {hurricane.type}</p>
                <p className="text-sm">Category: {hurricane.category || 'TS'}</p>
              </div>
            </Tooltip>
          </CircleMarker>
          
          {/* Forecast path polyline for selected or when "Show Forecast Paths" is enabled */}
          {(showForecastPaths || selectedHurricane?.id === hurricane.id) && (
            <Polyline
              positions={getPredictedPath(hurricane)}
              pathOptions={{
                color: getHurricaneColor(hurricane),
                weight: 2,
                opacity: selectedHurricane?.id === hurricane.id ? 0.9 : 0.6,
                dashArray: '5, 5'
              }}
            />
          )}
          
          {/* Forecast points along the path */}
          {selectedHurricane?.id === hurricane.id && 
            getPredictedPath(hurricane).map((point, index) => {
              if (index === 0) return null; // Skip the current position
              
              return (
                <CircleMarker
                  key={`forecast-${hurricane.id}-${index}`}
                  center={point}
                  radius={3} // Small dot for forecast points
                  pathOptions={{
                    color: '#ffffff',
                    fillColor: getHurricaneColor(hurricane),
                    fillOpacity: 0.8 - (index * 0.1), // Decreasing opacity for further predictions
                    weight: 1
                  }}
                >
                  <Tooltip 
                    direction="top" 
                    offset={[0, -3]} 
                    opacity={0.9}
                    permanent={false}
                  >
                    <div className="px-2 py-1">
                      <p className="text-sm font-bold">+{index * 24}h Forecast</p>
                    </div>
                  </Tooltip>
                </CircleMarker>
              );
            })
          }
        </React.Fragment>
      )
    ))}
      {/* Render potential storm formation areas */}
        {getVisiblePotentialStormAreas().map(area => (
          <CircleMarker
            key={`potential-${area.id}`}
            center={[area.position[0], area.position[1]]}
            radius={12} // These are larger than the other dots - to make it more clear that they are potential areas
            pathOptions={{
              color: '#ffffff',
              weight: 1,
              fillColor: area.probability > 0.6 ? '#FF5050' : 
                        area.probability > 0.4 ? '#FF9933' : '#FFDE33',
              fillOpacity: (animationFrame % 60) < 30 ? 0.4 : 0.8,
              className: 'animate-pulse' // Here I am using the animation class to make it pulse
            }}
          >
            <Tooltip
              direction="top"
              offset={[0, -5]}
              opacity={0.9}
            >
              <div className="px-2 py-1">
                <p className="font-bold">Potential Storm Formation</p>
                <p className="text-sm">Probability: {Math.round(area.probability * 100)}%</p>
                <p className="text-sm">Potential Intensity: {area.intensity}</p>
                <p className="text-sm">Basin: {area.basin}</p>
              </div>
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>
      
      {/* Control Panel */}
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
        
        {/* Layer Controls */}
        <div className="grid grid-cols-2 gap-1">
          {/* NASA Layers Button */}
          <button 
            onClick={toggleLayerControl}
            className={`flex items-center justify-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
              showLayerControl ? 'bg-[#2a3890] text-white' : 'bg-[#0b1021] text-gray-300'
            }`}
          >
            <Layers className="h-4 w-4" />
            <span className="hidden sm:inline">Layers</span>
          </button>
          
          {/* Search Button */}
          <button 
            onClick={toggleSearch}
            className={`flex items-center justify-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
              showSearch ? 'bg-[#2a3890] text-white' : 'bg-[#0b1021] text-gray-300'
            }`}
          >
            <Search className="h-4 w-4" />
            <span className="hidden sm:inline">Search</span>
          </button>
          
          {/* Filters Button */}
          <button 
            onClick={toggleFilters}
            className={`flex items-center justify-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
              showFilters ? 'bg-[#2a3890] text-white' : 'bg-[#0b1021] text-gray-300'
            }`}
          >
            <Filter className="h-4 w-4" />
            <span className="hidden sm:inline">Filter</span>
          </button>
          
          {/* Timeline Button */}
          <button 
            onClick={toggleTimeline}
            className={`flex items-center justify-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
              showTimeline ? 'bg-[#2a3890] text-white' : 'bg-[#0b1021] text-gray-300'
            }`}
          >
            <Clock className="h-4 w-4" />
            <span className="hidden sm:inline">Timeline</span>
          </button>
        </div>
        
        {/* Show Forecast Paths Button */}
        <button 
          onClick={toggleForecastPaths}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
            showForecastPaths ? 'bg-[#2a3890] text-white' : 'bg-[#0b1021] text-gray-300'
          }`}
        >
          {showForecastPaths ? (
            <span>Hide Forecast Paths</span>
          ) : (
            <span>Show Forecast Paths</span>
          )}
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
      
      {/* Search Panel */}
      {showSearch && (
        <div className="absolute top-24 right-2 bg-[#0b1021]/90 p-3 rounded-lg z-[1000] w-[250px]">
          <h4 className="text-sm font-bold mb-2 text-white">Search Storms</h4>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by name..."
            className="w-full p-2 bg-[#162040] text-white rounded border border-[#2a4858] text-sm"
          />
          
          {searchQuery && (
            <div className="mt-2">
              <p className="text-xs text-gray-300">Found {filteredHurricanes.length} results</p>
              
              {filteredHurricanes.length > 0 && (
                <div className="mt-2 max-h-[200px] overflow-y-auto">
                  {filteredHurricanes.map(hurricane => (
                    <button
                      key={hurricane.id}
                      onClick={() => {
                        onSelectHurricane(hurricane);
                        setShowSearch(false);
                      }}
                      className="w-full text-left p-2 rounded text-xs hover:bg-[#1a237e] mb-1 flex items-center"
                    >
                      <div 
                        className="w-3 h-3 rounded-full mr-2" 
                        style={{ backgroundColor: getHurricaneColor(hurricane) }}
                      ></div>
                      <span>{hurricane.name}</span>
                      <span className="ml-auto">{hurricane.category || 'TS'}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
      
      {/* Filters Panel */}
      {showFilters && (
        <div className="absolute top-24 right-2 bg-[#0b1021]/90 p-3 rounded-lg z-[1000] w-[250px]">
          <h4 className="text-sm font-bold mb-2 text-white">Filter Storms</h4>
          
          {/* Category Filter */}
          <div className="mb-3">
            <p className="text-xs text-gray-300 mb-1">Category</p>
            <div className="grid grid-cols-3 gap-1">
              {[0, 1, 2, 3, 4, 5, 'TS', 'TD'].map(category => (
                <button
                  key={category}
                  onClick={() => setFilterCategory(filterCategory === category ? null : category)}
                  className={`px-2 py-1 rounded text-xs transition-colors ${
                    filterCategory === category 
                      ? 'bg-[#1a237e] text-white' 
                      : 'bg-[#162040] text-gray-300'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>
          
          {/* Storm Type Filter */}
          <div className="mb-3">
            <p className="text-xs text-gray-300 mb-1">Storm Type</p>
            <div className="grid grid-cols-1 gap-1 max-h-[100px] overflow-y-auto">
              {stormTypes.map(type => (
                <button
                  key={type}
                  onClick={() => setFilterStormType(filterStormType === type ? null : type)}
                  className={`px-2 py-1 rounded text-xs transition-colors ${
                    filterStormType === type 
                      ? 'bg-[#1a237e] text-white' 
                      : 'bg-[#162040] text-gray-300'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>
          
          {/* Region Filter */}
          <div className="mb-3">
            <p className="text-xs text-gray-300 mb-1">Basin</p>
            <div className="grid grid-cols-2 gap-1">
              {['NA', 'EP', 'WP', 'NI', 'SI', 'SP'].map(basin => (
                <button
                  key={basin}
                  onClick={() => setFilterRegion(filterRegion === basin ? null : basin)}
                  className={`px-2 py-1 rounded text-xs transition-colors ${
                    filterRegion === basin 
                      ? 'bg-[#1a237e] text-white' 
                      : 'bg-[#162040] text-gray-300'
                  }`}
                >
                  {basin}
                </button>
              ))}
            </div>
          </div>
          
          {/* Clear Filters */}
          <button
            onClick={() => {
              setFilterCategory(null);
              setFilterRegion(null);
              setFilterStormType(null);
            }}
            className="w-full text-center p-2 rounded text-xs bg-[#3e1a1a] text-gray-300 hover:bg-[#641f1f] transition-colors"
          >
            Clear Filters
          </button>
        </div>
      )}
      
      {/* Timeline Slider */}
      {showTimeline && (
        <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 bg-[#0b1021]/90 p-3 rounded-lg z-[1000] w-[80%] max-w-[600px]">
          <h4 className="text-sm font-bold mb-2 text-white text-center">Storm Timeline</h4>
          
          <input
            type="range"
            min="0"
            max="100"
            value={currentTimelineValue}
            onChange={handleTimelineChange}
            className="w-full"
          />
          
          <div className="flex justify-between text-xs text-gray-300 mt-1">
            <span>-72h</span>
            <span>-48h</span>
            <span>-24h</span>
            <span>Now</span>
            <span>+24h</span>
            <span>+48h</span>
            <span>+72h</span>
          </div>
          <div className="text-center text-xs text-blue-400 mt-1">
            {timelineMode === 'present' ? 'Current' : timelineMode}
          </div>
        </div>
      )}
      
      {/* Legend */}
      <div className="absolute bottom-2 right-2 bg-[#0b1021]/70 text-white p-3 rounded-lg z-[1000]">
        <h4 className="text-xs font-bold mb-2">Storm Categories</h4>
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
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#9932CC' }}></div>
            <span>Thunderstorm</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#87CEFA' }}></div>
            <span>Winter Storm</span>
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
      
      {/* Filter/Search Indicator */}
      {(searchQuery || filterCategory !== null || filterRegion || filterStormType) && (
        <div className="absolute top-2 left-2 bg-[#1a237e] p-2 rounded-lg z-[1000]">
          <div className="flex items-center gap-2 text-xs">
            <Filter className="h-4 w-4 text-yellow-400" />
            <span>Filters Applied</span>
            <button 
              onClick={() => {
                setSearchQuery('');
                setFilterCategory(null);
                setFilterRegion(null);
                setFilterStormType(null);
              }}
              className="text-red-400 hover:text-red-300"
            >
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AtlasCommandMap;