"""
Service to collect and preprocess historical hurricane data for agent training
"""

import random
import math
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from .utils import haversine_distance, get_hurricane_category, safe_get

# Set up logging
logger = logging.getLogger('hurricane_agents.data')

# Cache for processed data to avoid repeated fetches
cached_hurricanes = None
data_load_promise = None

async def fetch_historical_hurricane_data(options: Dict = None) -> List[Dict]:
    """
    Fetch historical hurricane tracks from IBTrACS.
    
    Args:
        options: Options for data fetching
            - full_history: Whether to load full history or recent data
            - force_reload: Force reload of data even if cached
            - min_track_points: Minimum number of track points for a storm
            - min_category: Minimum storm category to include
            
    Returns:
        List of hurricane data dictionaries
    """
    global cached_hurricanes, data_load_promise
    
    if options is None:
        options = {}
    
    try:
        # If we're already loading data, return the existing promise
        if data_load_promise is not None:
            return await data_load_promise
        
        # If we have cached data and aren't forced to reload, use cache
        if cached_hurricanes is not None and not options.get('force_reload', False):
            logger.info('Using cached hurricane data')
            return cached_hurricanes
        
        # Set default options for hurricane data
        default_options = {
            'full_history': False,  # Use last3years by default for faster loading
            'min_track_points': 5,   # Ensure storms have enough data points
            'min_category': 0       # Include all storms by default
        }
        
        merged_options = {**default_options, **options}
        
        logger.info(f'Fetching historical hurricane data with options: {merged_options}')
        
        # In a real implementation, this would call the ibtracs connector
        # For now, just simulate data
        try:
            # This is where we would call the IBTrACS connector
            # data = await fetch_ibtracs_hurricanes(merged_options)
            
            # For development, use simulated data
            data = simulate_historical_data()
            
            logger.info(f'Loaded {len(data)} historical hurricanes')
            
            # Cache the data
            cached_hurricanes = data
            
            return data
        except Exception as e:
            logger.error(f'Error loading hurricane data: {e}')
            # Fall back to simulated data
            logger.warning('Falling back to simulated data')
            return simulate_historical_data()
            
    except Exception as e:
        logger.error(f'Error in fetch_historical_hurricane_data: {e}')
        return simulate_historical_data()

def simulate_historical_data() -> List[Dict]:
    """
    Simulate historical hurricane data for development (fallback).
    This is only used if real data loading fails.
    
    Returns:
        List of simulated hurricane data
    """
    logger.warning('Using simulated hurricane data')
    hurricanes = []
    years = 10
    hurricanes_per_year = 5
    
    for year in range(2015, 2015 + years):
        for i in range(hurricanes_per_year):
            hurricane_id = f'H{year}{i+1}'
            name = f'Hurricane {chr(65 + i)}'
            
            # Generate random duration between 5 and 14 days
            duration = 5 + random.randint(0, 9)
            
            # Generate random start time between June and September
            month = random.randint(5, 8)  # 5=June, 8=September
            day = random.randint(1, 28)
            start_time = pd.Timestamp(year, month, day)
            
            # Generate track data
            track = generate_hurricane_track(duration)
            
            hurricanes.append({
                'id': hurricane_id,
                'name': name,
                'year': year,
                'basin': 'NA',  # North Atlantic
                'start_time': start_time,
                'initial_position': track[0]['position'],
                'initial_wind_speed': track[0]['wind_speed'],
                'initial_pressure': track[0]['pressure'],
                'track': track
            })
    
    return hurricanes

def generate_hurricane_track(days: int) -> List[Dict]:
    """
    Generate a simulated hurricane track.
    
    Args:
        days: Duration in days
        
    Returns:
        List of track points
    """
    track = []
    time_steps = days * 4  # 4 observations per day (every 6 hours)
    
    # Starting position - random location in hurricane formation regions
    regions = [
        {'lat': 15, 'lon': -60},  # Caribbean
        {'lat': 20, 'lon': -45},  # Central Atlantic
        {'lat': 12, 'lon': -35}   # East Atlantic
    ]
    
    region = random.choice(regions)
    lat = region['lat'] + (random.random() * 4) - 2
    lon = region['lon'] + (random.random() * 4) - 2
    
    # Initial intensity
    wind_speed = 30 + random.random() * 20  # 30-50 mph
    pressure = 1010 - (wind_speed / 2)  # Roughly correlate with wind speed
    
    # Typical hurricane movement is NW in Atlantic
    lat_speed = 0.1 + random.random() * 0.2  # 0.1-0.3 degrees per time step
    lon_speed = -0.2 - random.random() * 0.3  # -0.2 to -0.5 degrees per time step
    
    for i in range(time_steps):
        timestamp = pd.Timestamp.now() + pd.Timedelta(hours=i * 6)
        
        # Add some random variation to movement
        lat_speed += (random.random() * 0.1) - 0.05
        lon_speed += (random.random() * 0.1) - 0.05
        
        # Constrain speeds to realistic values
        lat_speed = max(-0.3, min(0.5, lat_speed))
        lon_speed = max(-0.6, min(0.1, lon_speed))
        
        # Update position
        lat += lat_speed
        lon += lon_speed
        
        # Update intensity based on typical lifecycle
        # Intensification in first third, peak in middle, weakening in final third
        if i < time_steps / 3:
            wind_speed += (2 + random.random() * 5)  # Intensification
        elif i > (time_steps * 2) / 3:
            wind_speed -= (1 + random.random() * 6)  # Weakening
        else:
            wind_speed += (random.random() * 8) - 4  # Fluctuation near peak
        
        # Limit wind speed to realistic values
        wind_speed = max(30, min(175, wind_speed))
        
        # Pressure is inversely related to wind speed
        pressure = 1010 - (wind_speed / 2)
        
        track.append({
            'position': {'lat': lat, 'lon': lon},
            'wind_speed': wind_speed,
            'pressure': pressure,
            'timestamp': timestamp
        })
    
    return track

def preprocess_data_for_training(hurricanes: List[Dict]) -> List[Dict]:
    """
    Preprocess data for agent training.
    
    Args:
        hurricanes: List of hurricane data
        
    Returns:
        Preprocessed data
    """
    # Extract features and normalize
    return [preprocess_hurricane(hurricane) for hurricane in hurricanes]

def preprocess_hurricane(hurricane: Dict) -> Dict:
    """
    Preprocess a single hurricane's data.
    
    Args:
        hurricane: Hurricane data
        
    Returns:
        Preprocessed hurricane data
    """
    # Calculate the maximum wind speed for the hurricane
    wind_speeds = [safe_get(point, 'wind_speed', default=0) for point in hurricane.get('track', [])]
    max_wind_speed = max(wind_speeds) if wind_speeds else 0
    
    # Process track with derived features
    processed_track = []
    track = hurricane.get('track', [])
    
    for index, point in enumerate(track):
        # Skip first point for derivatives
        prev_point = track[index - 1] if index > 0 else point
        
        # Calculate time difference in hours
        time_diff_hours = 6  # Assume 6-hour intervals if no timestamps
        if 'timestamp' in point and 'timestamp' in prev_point:
            try:
                current_time = pd.Timestamp(point['timestamp'])
                prev_time = pd.Timestamp(prev_point['timestamp'])
                time_diff_hours = (current_time - prev_time).total_seconds() / 3600
            except:
                # Use default if timestamp conversion fails
                pass
        
        # Position change rate (degrees per hour)
        lat_change_rate = 0
        lon_change_rate = 0
        if time_diff_hours > 0:
            lat_current = safe_get(point, 'position', 'lat', default=0)
            lat_prev = safe_get(prev_point, 'position', 'lat', default=0)
            lon_current = safe_get(point, 'position', 'lon', default=0)
            lon_prev = safe_get(prev_point, 'position', 'lon', default=0)
            
            lat_change_rate = (lat_current - lat_prev) / time_diff_hours
            lon_change_rate = (lon_current - lon_prev) / time_diff_hours
        
        # Intensity change rate (mph per hour)
        wind_speed_change_rate = 0
        if time_diff_hours > 0:
            wind_current = safe_get(point, 'wind_speed', default=0)
            wind_prev = safe_get(prev_point, 'wind_speed', default=0)
            wind_speed_change_rate = (wind_current - wind_prev) / time_diff_hours
        
        # Pressure change rate (mb per hour)
        pressure_change_rate = 0
        if time_diff_hours > 0:
            pressure_current = safe_get(point, 'pressure', default=1000)
            pressure_prev = safe_get(prev_point, 'pressure', default=1000)
            pressure_change_rate = (pressure_current - pressure_prev) / time_diff_hours
        
        # Normalize wind speed relative to maximum
        normalized_wind_speed = (point.get('wind_speed', 0) / max_wind_speed) if max_wind_speed > 0 else 0
        
        # Calculate distance from genesis (initial position)
        distance_from_genesis = haversine_distance(
            hurricane.get('initial_position', {}),
            point.get('position', {})
        )
        
        # Time since genesis (in hours)
        hours_since_genesis = 0
        if 'timestamp' in point and 'start_time' in hurricane:
            try:
                point_time = pd.Timestamp(point['timestamp'])
                start_time = pd.Timestamp(hurricane['start_time'])
                hours_since_genesis = (point_time - start_time).total_seconds() / 3600
            except:
                # Use default if timestamp conversion fails
                hours_since_genesis = index * 6  # Assume 6-hour intervals
        else:
            hours_since_genesis = index * 6  # Assume 6-hour intervals
        
        # Phase of storm lifecycle (early, peak, late)
        storm_phase = get_storm_phase(index, len(track))
        
        processed_point = {
            **point,
            # Original derived features
            'wind_speed_change': point.get('wind_speed', 0) - hurricane.get('initial_wind_speed', 0),
            'pressure_change': point.get('pressure', 0) - hurricane.get('initial_pressure', 0),
            
            # New derived features
            'lat_change_rate': lat_change_rate,
            'lon_change_rate': lon_change_rate,
            'wind_speed_change_rate': wind_speed_change_rate,
            'pressure_change_rate': pressure_change_rate,
            'normalized_wind_speed': normalized_wind_speed,
            'distance_from_genesis': distance_from_genesis,
            'hours_since_genesis': hours_since_genesis,
            
            # Phase of storm lifecycle (early, peak, late)
            'storm_phase': storm_phase
        }
        processed_track.append(processed_point)
    
    return {
        **hurricane,
        'max_wind_speed': max_wind_speed,
        'track': processed_track
    }

def get_storm_phase(index: int, total_length: int) -> str:
    """
    Get the phase of a storm based on position in track.
    
    Args:
        index: Current index
        total_length: Total track length
        
    Returns:
        Storm phase ("early", "peak", or "late")
    """
    if total_length <= 0:
        return "unknown"
        
    normalized_position = index / total_length
    
    if normalized_position < 0.3:
        return "early"    # Formation/Intensification
    if normalized_position < 0.7:
        return "peak"     # Mature/Peak
    return "late"         # Weakening/Dissipation