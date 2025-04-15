"""
Utility functions for hurricane prediction agents
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hurricane_agents')


def haversine_distance(pos1: Dict, pos2: Dict) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    
    Args:
        pos1: First position with latitude and longitude
        pos2: Second position with latitude and longitude
        
    Returns:
        Distance in kilometers
    """
    if not pos1 or not pos2:
        return 1000.0  # Large error if positions missing
        
    # Convert decimal degrees to radians
    lat1 = math.radians(pos1.get('lat', 0))
    lon1 = math.radians(pos1.get('lon', 0))
    lat2 = math.radians(pos2.get('lat', 0))
    lon2 = math.radians(pos2.get('lon', 0))
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r


def determine_basin(lon: float, lat: float) -> str:
    """
    Determine the ocean basin based on coordinates.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        
    Returns:
        Basin identifier string
    """
    # Simple basin determination based on coordinates
    if lon > -100 and lon < 0 and lat > 0:
        return 'NA'  # North Atlantic
    elif lon >= -180 and lon < -100 and lat > 0:
        return 'EP'  # Eastern Pacific
    elif lon >= 100 and lon < 180 and lat > 0:
        return 'WP'  # Western Pacific
    elif lon >= 40 and lon < 100 and lat > 0:
        return 'NI'  # North Indian
    elif lon >= 40 and lon < 135 and lat <= 0:
        return 'SI'  # South Indian
    elif lon >= 135 and lon < 180 and lat <= 0:
        return 'SP'  # South Pacific
    
    return 'UNKNOWN'


def wind_pressure_relationship(wind_speed: float = None, pressure: float = None) -> Dict:
    """
    Calculate the relationship between wind speed and pressure.
    Provide either wind_speed or pressure to estimate the other.
    
    Args:
        wind_speed: Wind speed in mph (optional)
        pressure: Pressure in millibars (optional)
        
    Returns:
        Dictionary with both wind_speed and pressure
    """
    if wind_speed is not None:
        # Estimate pressure from wind speed
        # P = 1010 - (wind_speed/1.15)^2/100
        estimated_pressure = 1010 - (wind_speed/1.15)**2/100
        return {
            'wind_speed': wind_speed,
            'pressure': max(880, min(1020, estimated_pressure))
        }
    elif pressure is not None:
        # Estimate wind speed from pressure
        # wind_speed = 1.15 * sqrt((1010 - pressure) * 100)
        delta_pressure = max(0, 1010 - pressure)  # Ensure non-negative
        estimated_wind_speed = 1.15 * math.sqrt(delta_pressure * 100)
        return {
            'wind_speed': estimated_wind_speed,
            'pressure': pressure
        }
    else:
        # Default values if neither is provided
        return {
            'wind_speed': 0,
            'pressure': 1010
        }


def get_hurricane_category(wind_speed: float) -> str:
    """
    Get hurricane category from wind speed in mph.
    
    Args:
        wind_speed: Wind speed in mph
        
    Returns:
        Category as string ('TD', 'TS', '1'-'5')
    """
    if wind_speed < 39:
        return 'TD'  # Tropical Depression
    elif wind_speed < 74:
        return 'TS'  # Tropical Storm
    elif wind_speed < 96:
        return '1'
    elif wind_speed < 111:
        return '2'
    elif wind_speed < 130:
        return '3'
    elif wind_speed < 157:
        return '4'
    else:
        return '5'


def category_to_numeric(category: str) -> int:
    """
    Convert hurricane category to numeric value.
    
    Args:
        category: Hurricane category as string
        
    Returns:
        Numeric value (0-5)
    """
    if category == 'TD':
        return 0
    elif category == 'TS':
        return 0
    else:
        try:
            return int(category)
        except (ValueError, TypeError):
            return 0


def timestamp_to_datetime(timestamp: Any) -> pd.Timestamp:
    """
    Convert various timestamp formats to pandas Timestamp.
    
    Args:
        timestamp: Timestamp in various formats
        
    Returns:
        pandas Timestamp object
    """
    if timestamp is None:
        return pd.Timestamp.now()
    
    if isinstance(timestamp, pd.Timestamp):
        return timestamp
    
    try:
        return pd.Timestamp(timestamp)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert timestamp: {timestamp}")
        return pd.Timestamp.now()


def create_ensemble_prediction_points(
    base_prediction: Dict, 
    num_members: int = 10,
    time_days: int = 5,
    time_step_hours: int = 6
) -> List[Dict]:
    """
    Create ensemble prediction points from a base prediction.
    
    Args:
        base_prediction: Base prediction
        num_members: Number of ensemble members
        time_days: Number of days to predict
        time_step_hours: Time step in hours
        
    Returns:
        List of prediction points for ensemble members
    """
    ensemble_points = []
    
    # Extract base values
    base_lat = base_prediction.get('position', {}).get('lat', 25.0)
    base_lon = base_prediction.get('position', {}).get('lon', -75.0)
    base_wind = base_prediction.get('wind_speed', 75.0)
    base_pressure = base_prediction.get('pressure', 990.0)
    
    # Create ensemble variations
    for member in range(num_members):
        # Each member has a slightly different bias
        lat_bias = np.random.normal(0, 0.5)  # Geographic bias in position
        lon_bias = np.random.normal(0, 0.5)
        intensity_bias = np.random.normal(0, 5.0)  # Intensity bias in mph
        
        track_points = []
        
        # Starting point (current conditions)
        current_lat = base_lat
        current_lon = base_lon
        current_wind = base_wind
        current_pressure = base_pressure
        
        for hour in range(0, time_days * 24 + 1, time_step_hours):
            day = hour // 24
            
            # Calculate hour-specific uncertainty
            # Uncertainty grows with forecast time
            hour_factor = hour / 24  # Convert to days
            position_spread = 0.1 * hour_factor
            intensity_spread = 3.0 * hour_factor
            
            # Add randomness that accumulates over time
            random_lat = current_lat + lat_bias * hour_factor + np.random.normal(0, position_spread)
            random_lon = current_lon + lon_bias * hour_factor + np.random.normal(0, position_spread)
            random_wind = current_wind + intensity_bias * hour_factor + np.random.normal(0, intensity_spread)
            
            # Ensure wind speed is not negative
            random_wind = max(0, random_wind)
            
            # Calculate pressure from wind speed for consistency
            relationship = wind_pressure_relationship(wind_speed=random_wind)
            random_pressure = relationship['pressure']
            
            # Add some persistence to changes
            current_lat = random_lat
            current_lon = random_lon
            current_wind = random_wind
            current_pressure = random_pressure
            
            track_points.append({
                'hour': hour,
                'day': day,
                'position': {'lat': random_lat, 'lon': random_lon},
                'wind_speed': random_wind,
                'pressure': random_pressure,
                'category': get_hurricane_category(random_wind),
                'ensemble_member': member
            })
        
        ensemble_points.extend(track_points)
    
    return ensemble_points


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if the denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def safe_get(dictionary: Dict, *keys, default=None) -> Any:
    """
    Safely get a nested value from a dictionary.
    
    Args:
        dictionary: Dictionary to get value from
        *keys: Keys to traverse
        default: Default value if key not found
        
    Returns:
        Value at key path or default
    """
    if dictionary is None:
        return default
    
    current = dictionary
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current