"""
OpenMeteo Connector for global hurricane and severe weather data

This module provides functions to interact with the OpenMeteo API
and supplements the NOAA/IBTRACS data for non-US regions.
"""

import aiohttp
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger('hurricane_agents.openmeteo')

# Base URLs
OPENMETEO_FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
OPENMETEO_MARINE_URL = 'https://marine-api.open-meteo.com/v1/marine'

# Ocean basin identifiers mapped to coordinates
BASIN_COORDINATES = {
    'WP': {'name': 'Western Pacific', 'hotspots': [
        {'lat': 15.0, 'lon': 130.0, 'name': 'Philippine Sea'},
        {'lat': 20.0, 'lon': 135.0, 'name': 'Western Pacific'},
        {'lat': 10.0, 'lon': 125.0, 'name': 'Philippines Region'}
    ]},
    'EP': {'name': 'Eastern Pacific', 'hotspots': [
        {'lat': 15.0, 'lon': -105.0, 'name': 'Eastern Pacific'},
        {'lat': 12.0, 'lon': -120.0, 'name': 'Central Pacific'}
    ]},
    'NA': {'name': 'North Atlantic', 'hotspots': [
        {'lat': 25.0, 'lon': -75.0, 'name': 'Western Atlantic'},
        {'lat': 15.0, 'lon': -55.0, 'name': 'Central Atlantic'},
        {'lat': 20.0, 'lon': -85.0, 'name': 'Caribbean Sea'}
    ]},
    'NI': {'name': 'North Indian', 'hotspots': [
        {'lat': 15.0, 'lon': 85.0, 'name': 'Bay of Bengal'},
        {'lat': 15.0, 'lon': 65.0, 'name': 'Arabian Sea'}
    ]},
    'SI': {'name': 'South Indian', 'hotspots': [
        {'lat': -15.0, 'lon': 70.0, 'name': 'Madagascar Basin'},
        {'lat': -15.0, 'lon': 90.0, 'name': 'Eastern Indian Ocean'}
    ]},
    'SP': {'name': 'South Pacific', 'hotspots': [
        {'lat': -15.0, 'lon': 170.0, 'name': 'Fiji Basin'},
        {'lat': -15.0, 'lon': 145.0, 'name': 'Coral Sea'}
    ]}
}

# Storm classification thresholds (wind speeds in km/h)
STORM_THRESHOLDS = {
    'TD': {'wind_speed': 63, 'pressure': 1000},  # Tropical Depression
    'TS': {'wind_speed': 118, 'pressure': 990},  # Tropical Storm
    'CAT1': {'wind_speed': 154, 'pressure': 980},  # Category 1
    'CAT2': {'wind_speed': 177, 'pressure': 965},  # Category 2
    'CAT3': {'wind_speed': 209, 'pressure': 945},  # Category 3
    'CAT4': {'wind_speed': 252, 'pressure': 920},  # Category 4
    'CAT5': {'wind_speed': 252, 'pressure': 920},  # Category 5
}

# Cache for API responses (to avoid repeated calls)
response_cache = {}
CACHE_DURATION = 30 * 60  # 30 minutes in seconds

async def fetch_open_meteo_data(lat: float, lon: float, basin: str = "GLOBAL", include_marine: bool = True) -> Dict:
    """
    Fetch comprehensive weather data from OpenMeteo APIs for a specific location
    
    Args:
        lat: Latitude
        lon: Longitude
        basin: Ocean basin identifier (WP, EP, NA, NI, SI, SP, or GLOBAL)
        include_marine: Whether to include marine data (wave height, etc.)
        
    Returns:
        Combined weather and marine data
    """
    # Create cache key
    cache_key = f"{lat}_{lon}_{basin}_{include_marine}"
    current_time = datetime.now().timestamp()
    
    # Check cache first
    if cache_key in response_cache:
        cache_data, cache_time = response_cache[cache_key]
        if current_time - cache_time < CACHE_DURATION:
            logger.info(f"Using cached OpenMeteo data for {lat}, {lon}")
            return cache_data
    
    try:
        # Prepare parameters for main weather API
        forecast_params = {
            'latitude': str(lat),
            'longitude': str(lon),
            'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_gusts_10m,wind_direction_10m,precipitation',
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max',
            'timezone': 'auto',
            'forecast_days': '7'
        }
        
        # Make API requests concurrently
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_forecast(session, forecast_params)]
            
            # Add marine API request if requested
            if include_marine:
                marine_params = {
                    'latitude': str(lat),
                    'longitude': str(lon),
                    'hourly': 'wave_height,wave_direction,wave_period,wind_wave_height,swell_wave_height',  # Remove wind_speed_10m
                    'timezone': 'auto',
                    'forecast_days': '7'
                }
                tasks.append(fetch_marine(session, marine_params))
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        forecast_data = results[0] if not isinstance(results[0], Exception) else {}
        marine_data = results[1] if include_marine and not isinstance(results[1], Exception) else {}
        
        # Combine data
        combined_data = {
            'forecast': forecast_data,
            'marine': marine_data if include_marine else None,
            'basin': basin,
            'source': 'OpenMeteo'
        }
        
        # Process into a standardized format
        processed_data = process_openmeteo_data(combined_data)
        
        # Cache the result
        response_cache[cache_key] = (processed_data, current_time)
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error fetching OpenMeteo data: {e}")
        return {
            'error': True,
            'message': str(e),
            'basin': basin
        }

async def fetch_forecast(session: aiohttp.ClientSession, params: Dict) -> Dict:
    """Fetch data from the OpenMeteo forecast API"""
    try:
        async with session.get(OPENMETEO_FORECAST_URL, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"OpenMeteo API error: {response.status} - {error_text}")
                raise Exception(f"OpenMeteo API error: {response.status}")
                
            return await response.json()
    except Exception as e:
        logger.error(f"Error in fetch_forecast: {e}")
        raise e

async def fetch_marine(session: aiohttp.ClientSession, params: Dict) -> Dict:
    """Fetch data from the OpenMeteo marine API"""
    try:
        async with session.get(OPENMETEO_MARINE_URL, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"OpenMeteo Marine API error: {response.status} - {error_text}")
                raise Exception(f"OpenMeteo Marine API error: {response.status}")
                
            return await response.json()
    except Exception as e:
        logger.error(f"Error in fetch_marine: {e}")
        raise e

def process_openmeteo_data(data: Dict) -> Dict:
    """Process OpenMeteo data into a standardized format for the RL agent"""
    try:
        if 'error' in data:
            return data
            
        forecast = data.get('forecast', {})
        marine = data.get('marine', {})
        basin = data.get('basin', 'GLOBAL')
        
        # Extract hourly data
        hourly_data = []
        
        if 'hourly' in forecast:
            hourly = forecast['hourly']
            times = hourly.get('time', [])
            
            for i in range(len(times)):
                hourly_point = {
                    'time': times[i] if i < len(times) else None,
                    'temperature': hourly.get('temperature_2m', [])[i] if i < len(hourly.get('temperature_2m', [])) else None,
                    'humidity': hourly.get('relative_humidity_2m', [])[i] if i < len(hourly.get('relative_humidity_2m', [])) else None,
                    'pressure': hourly.get('pressure_msl', [])[i] if i < len(hourly.get('pressure_msl', [])) else None,
                    'wind_speed': hourly.get('wind_speed_10m', [])[i] if i < len(hourly.get('wind_speed_10m', [])) else None,
                    'wind_gusts': hourly.get('wind_gusts_10m', [])[i] if i < len(hourly.get('wind_gusts_10m', [])) else None,
                    'wind_direction': hourly.get('wind_direction_10m', [])[i] if i < len(hourly.get('wind_direction_10m', [])) else None,
                    'precipitation': hourly.get('precipitation', [])[i] if i < len(hourly.get('precipitation', [])) else None,
                }
                
                # Add marine data if available
                if marine and 'hourly' in marine:
                    marine_hourly = marine['hourly']
                    hourly_point.update({
                        'wave_height': marine_hourly.get('wave_height', [])[i] if i < len(marine_hourly.get('wave_height', [])) else None,
                        'wave_direction': marine_hourly.get('wave_direction', [])[i] if i < len(marine_hourly.get('wave_direction', [])) else None,
                        'wave_period': marine_hourly.get('wave_period', [])[i] if i < len(marine_hourly.get('wave_period', [])) else None,
                    })
                
                hourly_data.append(hourly_point)
        
        # Determine if this is a tropical cyclone based on wind speed and pressure
        max_wind_speed = max([h.get('wind_speed', 0) or 0 for h in hourly_data], default=0)
        min_pressure = min([h.get('pressure', 1020) or 1020 for h in hourly_data], default=1020)
        
        # Convert km/h to mph for consistency with US data
        max_wind_speed_mph = max_wind_speed * 0.621371
        
        # Determine storm category
        category = get_storm_category(max_wind_speed, min_pressure)
        
        # Create a standardized observation format (matching NOAA format as closely as possible)
        observations = {
            'timestamp': hourly_data[0].get('time') if hourly_data else None,
            'temperature': hourly_data[0].get('temperature') if hourly_data else None,
            'windSpeed': max_wind_speed_mph,  # Convert to mph to match NOAA
            'windDirection': hourly_data[0].get('wind_direction') if hourly_data else None,
            'barometricPressure': min_pressure,
            'relativeHumidity': hourly_data[0].get('humidity') if hourly_data else None,
            'waveHeight': hourly_data[0].get('wave_height') if hourly_data else None,
            'dataSource': 'OpenMeteo'
        }
        
        # Generate forecast data in standardized format
        forecast_data = []
        
        time_steps = [0, 6, 12, 18, 24, 36, 48, 72, 96, 120]  # Hours ahead
        for hour in time_steps:
            if hour < len(hourly_data):
                point = hourly_data[hour]
                wind_speed_mph = (point.get('wind_speed') or 0) * 0.621371  # Convert to mph
                
                forecast_data.append({
                    'hour': hour,
                    'day': hour // 24 + 1,
                    'windSpeed': wind_speed_mph,
                    'pressure': point.get('pressure'),
                    'category': get_storm_category(point.get('wind_speed') or 0, point.get('pressure') or 1010),
                    'position': {'lat': forecast.get('latitude'), 'lon': forecast.get('longitude')},
                    'confidence': max(20, 100 - (hour * 0.6))  # Decreasing confidence with time
                })
        
        # Final processed result
        return {
            'observations': observations,
            'forecast': forecast_data,
            'riskLevel': determine_risk_level(max_wind_speed_mph, min_pressure, category),
            'basin': basin,
            'category': category,
            'rawData': {
                'hourly': hourly_data,
                'daily': forecast.get('daily')
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing OpenMeteo data: {e}")
        return {
            'error': True,
            'message': f"Error processing data: {str(e)}",
            'basin': data.get('basin', 'GLOBAL')
        }

def get_storm_category(wind_speed_kmh: float, pressure: float) -> str:
    """
    Determine storm category based on wind speed (km/h) and pressure (hPa)
    
    Returns the category as a string (TD, TS, 1, 2, 3, 4, 5)
    """
    # First check by wind speed (primary indicator)
    if wind_speed_kmh >= STORM_THRESHOLDS['CAT5']['wind_speed']:
        return "5"
    elif wind_speed_kmh >= STORM_THRESHOLDS['CAT4']['wind_speed']:
        return "4"
    elif wind_speed_kmh >= STORM_THRESHOLDS['CAT3']['wind_speed']:
        return "3"
    elif wind_speed_kmh >= STORM_THRESHOLDS['CAT2']['wind_speed']:
        return "2"
    elif wind_speed_kmh >= STORM_THRESHOLDS['CAT1']['wind_speed']:
        return "1"
    elif wind_speed_kmh >= STORM_THRESHOLDS['TS']['wind_speed']:
        return "TS"
    elif wind_speed_kmh >= STORM_THRESHOLDS['TD']['wind_speed']:
        return "TD"
    
    # If wind speed is below TD threshold, check pressure as a secondary indicator
    if pressure and pressure < STORM_THRESHOLDS['TS']['pressure']:
        return "TS"
    elif pressure and pressure < STORM_THRESHOLDS['TD']['pressure']:
        return "TD"
    
    # Not a classified storm
    return "0"

def determine_risk_level(wind_speed_mph: float, pressure: float, category: str) -> str:
    """Determine risk level based on storm characteristics"""
    # Simple risk assessment based on category
    if category == "4" or category == "5":
        return "extreme"
    elif category == "3":
        return "high"
    elif category == "1" or category == "2":
        return "moderate"
    elif category == "TS":
        return "low"
    else:
        return "minimal"

async def get_active_hurricanes_by_region(region: str = "GLOBAL") -> List[Dict]:
    """
    Get active severe weather events globally or by region
    Polls hotspots for each region to find severe storms
    
    Args:
        region: Optional region code to filter results (WP, EP, NA, NI, SI, SP, or GLOBAL)
        
    Returns:
        List of active storms in standardized format
    """
    try:
        all_storms = []
        
        # Determine which regions to check
        regions_to_check = [region] if region != "GLOBAL" else BASIN_COORDINATES.keys()
        
        # For each region, check all hotspots
        for region_code in regions_to_check:
            if region_code not in BASIN_COORDINATES:
                continue
                
            region_data = BASIN_COORDINATES[region_code]
            hotspots = region_data['hotspots']
            
            # Check each hotspot in parallel
            tasks = []
            for hotspot in hotspots:
                task = fetch_open_meteo_data(
                    hotspot['lat'], 
                    hotspot['lon'], 
                    region_code
                )
                tasks.append(task)
            
            # Await all hotspot checks
            results = await asyncio.gather(*tasks)
            
            # Process results to find storms
            for i, result in enumerate(results):
                if 'error' in result:
                    continue
                    
                hotspot = hotspots[i]
                
                # Check if this is a storm (category > 0)
                category = result.get('category')
                if category in ['TD', 'TS', '1', '2', '3', '4', '5']:
                    # Generate a unique ID
                    storm_id = f"openmeteo-{region_code}-{hotspot['name'].replace(' ', '-')}-{datetime.now().strftime('%Y%m%d')}"
                    
                    # Create a storm object in the standard format
                    storm = {
                        'id': storm_id,
                        'name': f"{region_data['name']} {category}",
                        'type': get_storm_type(category, region_code),
                        'category': category,
                        'basin': region_code,
                        'coordinates': [hotspot['lon'], hotspot['lat']],  # [lon, lat]
                        'windSpeed': result.get('observations', {}).get('windSpeed'),
                        'pressure': result.get('observations', {}).get('barometricPressure'),
                        'dataSource': 'OpenMeteo',
                        'severity': get_severity(category),
                        'certainty': 'Observed'
                    }
                    
                    all_storms.append(storm)
        
        return all_storms
    
    except Exception as e:
        logger.error(f"Error getting active hurricanes by region: {e}")
        return []

def get_storm_type(category: str, region: str) -> str:
    """Get the appropriate storm type name based on category and region"""
    # Different regions use different names for tropical cyclones
    if region == "WP":
        base_name = "Typhoon"
    elif region in ["NI", "SI", "SP"]:
        base_name = "Cyclone"
    else:  # NA, EP
        base_name = "Hurricane"
    
    # Category determines the exact type
    if category in ["1", "2", "3", "4", "5"]:
        return base_name
    elif category == "TS":
        return "Tropical Storm"
    else:  # TD
        return "Tropical Depression"

def get_severity(category: str) -> str:
    """Convert category to severity level for consistency with NOAA data"""
    if category in ["4", "5"]:
        return "Extreme"
    elif category == "3":
        return "Severe"
    elif category in ["1", "2"]:
        return "Moderate"
    elif category == "TS":
        return "Minor"
    else:  # TD or lower
        return "Minimal"