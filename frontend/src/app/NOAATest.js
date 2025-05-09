'use client';

import React, { useState, useEffect } from 'react';

export default function NOAATest() {
  const [apiData, setApiData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function testAPI() {
      try {
        setLoading(true);
        const response = await fetch('https://api.weather.gov/alerts/active', {
          headers: {
            'User-Agent': 'Atlas Command Center Test'
          }
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        setApiData(data);
        setError(null);
      } catch (err) {
        setError(err.message);
        setApiData(null);
      } finally {
        setLoading(false);
      }
    }

    testAPI();
  }, []);

  return (
    <div className="p-6 bg-[#0B1021] text-white min-h-screen">
      <h1 className="text-2xl font-bold mb-4">NOAA API Test</h1>
      
      {loading && (
        <div className="text-blue-400">Loading...</div>
      )}

      {error && (
        <div className="bg-red-900/50 p-4 rounded-lg mb-4">
          <h2 className="text-xl font-bold text-red-400 mb-2">Error</h2>
          <p>{error}</p>
        </div>
      )}

      {apiData && (
        <div className="space-y-4">
          <div className="bg-[#1a237e] p-4 rounded-lg">
            <h2 className="text-xl font-bold mb-2">Active Alerts</h2>
            <p>Total Alerts: {apiData.features?.length || 0}</p>
          </div>

          <div className="bg-[#1a237e] p-4 rounded-lg">
            <h2 className="text-xl font-bold mb-2">First Alert Details</h2>
            {apiData.features?.[0] ? (
              <div className="space-y-2">
                <p><span className="text-gray-400">Event:</span> {apiData.features[0].properties.event}</p>
                <p><span className="text-gray-400">Headline:</span> {apiData.features[0].properties.headline}</p>
                <p><span className="text-gray-400">Severity:</span> {apiData.features[0].properties.severity}</p>
              </div>
            ) : (
              <p>No alerts found</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}