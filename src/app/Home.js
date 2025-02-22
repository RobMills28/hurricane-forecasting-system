'use client';

import React, { useState, useEffect } from 'react';
import { AlertTriangle, Flame, Wind, CloudRain, Waves, Cloud } from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Custom marker icon
const customIcon = L.divIcon({
  className: 'custom-marker',
  html: `<div class="w-6 h-6 bg-red-500 rounded-full animate-pulse border-2 border-white flex items-center justify-center">
    <div class="w-4 h-4 bg-red-600 rounded-full"></div>
  </div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
  popupAnchor: [0, -12],
});

const DARK_MAP_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

const weatherEvents = [
  { id: 'wildfires', name: 'Wildfires', icon: Flame, color: 'text-orange-500' },
  { id: 'tornados', name: 'Tornados', icon: Wind, color: 'text-gray-400' },
  { id: 'hurricanes', name: 'Hurricanes', icon: CloudRain, color: 'text-blue-500' },
  { id: 'floods', name: 'Floods', icon: Waves, color: 'text-blue-400' },
  { id: 'storms', name: 'Storms', icon: Cloud, color: 'text-gray-300' }
];

export default function Home() {
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const [activeEvent, setActiveEvent] = useState(null);
  const [selectedEventType, setSelectedEventType] = useState('drought');

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const centralAsiaPosition = [45.0, 68.0];

  return (
    <main className="min-h-screen bg-[#0B1021] text-white">
      {/* Alert Banner */}
      <div className="w-full bg-[#1a237e] px-4 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-400" />
            <span className="font-mono font-bold">
              WARNING: Severe Drought Conditions in Central Asia
            </span>
          </div>
          <span className="text-gray-400 font-mono">{currentTime}</span>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Atlas Command Center</h1>
          
          {/* Navigation Menu */}
          <div className="flex gap-4">
            {weatherEvents.map((event) => (
              <button
                key={event.id}
                onClick={() => setSelectedEventType(event.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors
                  ${selectedEventType === event.id ? 'bg-[#1a237e]' : 'hover:bg-[#1a237e]/50'}`}
              >
                <event.icon className={`h-5 w-5 ${event.color}`} />
                <span>{event.name}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Map Container */}
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2 h-[70vh] bg-[#0d1424] rounded-lg overflow-hidden">
            <MapContainer
              center={centralAsiaPosition}
              zoom={4}
              className="w-full h-full"
              style={{ background: '#0d1424' }}
              minZoom={2}  // Restrict minimum zoom level
              maxBounds={[[-90, -180], [90, 180]]}  // Restrict map bounds
              maxBoundsViscosity={1.0}  // Make bounds restrictions solid
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url={DARK_MAP_URL}
                noWrap={true}  // Prevent tile repetition
              />
              <Marker position={centralAsiaPosition} icon={customIcon}>
                <Popup className="custom-popup">
                  <div className="bg-[#1a237e] text-white p-4 rounded-lg shadow-lg min-w-[200px]">
                    <h3 className="font-bold text-lg mb-2 flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5 text-yellow-400" />
                      Severe Drought
                    </h3>
                    <div className="space-y-2">
                      <p><span className="text-gray-400">Region:</span> Central Asia</p>
                      <p><span className="text-gray-400">Severity:</span> High</p>
                      <p><span className="text-gray-400">Duration:</span> 3 months</p>
                    </div>
                  </div>
                </Popup>
              </Marker>
            </MapContainer>
          </div>

          {/* Side Panel */}
          <div className="h-[70vh] bg-[#0d1424] rounded-lg p-4 overflow-y-auto">
            <h2 className="text-xl font-bold mb-4">Event Details</h2>
            {activeEvent ? (
              <div className="space-y-4">
                {/* Event details content */}
              </div>
            ) : (
              <p className="text-gray-400">Select an event marker to view details</p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}