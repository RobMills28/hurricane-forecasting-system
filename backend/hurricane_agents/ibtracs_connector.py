"""
Connector for IBTrACS (International Best Track Archive for Climate Stewardship) data

This module handles downloading, parsing, and processing global tropical cyclone data
"""

import pandas as pd
import numpy as np
import aiohttp
import io
import logging
import math
from typing import Dict, List, Optional, Any, Union
from .utils import get_hurricane_category, determine_basin

# Set up logging
logger = logging.getLogger('hurricane_agents.ibtracs')

# Configuration
IBTRACS_BASE_URL = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv'
IBTRACS_SINCE1980_URL = f'{IBTRACS_BASE_URL}/ibtracs.since1980.list.v04r00.csv'
IBTRACS_LAST3YEARS_URL = f'{IBTRACS_BASE_URL}/ibtracs.last3years.list.v04r00.csv'

async def fetch_ibtracs_data(dataset: str = 'since1980') -> bytes:
    """
    Fetches IBTrACS data.
    
    Args:
        dataset: Dataset type: 'since1980' or 'last3years'
        
    Returns:
        Raw CSV data as bytes
    """
    url = IBTRACS_LAST3YEARS_URL if dataset == 'last3years' else IBTRACS_SINCE1980_URL
    logger.info(f'Fetching IBTrACS data from: {url}')
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f'Failed to fetch IBTrACS data: {response.status} {response.reason}')
                return await response.read()
    except Exception as e:
        logger.error(f'Error fetching IBTrACS data: {e}')
        raise e

async def parse_ibtracs_data(data: bytes) -> List[Dict]:
    """
    Parses IBTrACS CSV data.
    
    Args:
        data: Raw CSV data
        
    Returns:
        Parsed cyclone records
    """
    try:
        # Read CSV data
        df = pd.read_csv(io.BytesIO(data), 
                         header=0, 
                         dtype={'SID': str})
        
        # Convert to list of dictionaries
        records = df.to_dict('records')
        logger.info(f'Parsed {len(records)} IBTrACS records')
        return records
    except Exception as e:
        logger.error(f'Error parsing CSV: {e}')
        raise e

def process_ibtracs_records(records: List[Dict]) -> List[Dict]:
    """
    Processes raw IBTrACS records into standardised hurricane data.
    
    Args:
        records: Raw IBTrACS records
        
    Returns:
        Processed hurricane data
    """
    # Group records by storm ID (each storm has multiple time points)
    storm_groups = {}
    
    for record in records:
        # Skip records with missing key data
        if not record.get('SID') or \
           record.get('LAT') is None or record.get('LON') is None or \
           record.get('USA_WIND') is None or record.get('USA_PRES') is None:
            continue
        
        # Use SID (Storm ID) as the grouping key
        storm_id = record['SID']
        
        if storm_id not in storm_groups:
            storm_groups[storm_id] = []
        
        # Convert ISO time format
        timestamp = None
        try:
            if record.get('ISO_TIME'):
                timestamp = pd.Timestamp(record['ISO_TIME'])
            else:
                # Fallback if ISO_TIME is missing - construct from year/month/day/hour
                timestamp = pd.Timestamp(
                    year=record.get('SEASON', 2000),
                    month=record.get('MONTH', 1),
                    day=record.get('DAY', 1),
                    hour=record.get('HOUR', 0)
                )
        except Exception as e:
            logger.warning(f'Invalid date for storm {storm_id}: {e}')
            timestamp = pd.Timestamp.now()  # Fallback to prevent crashes
        
        # Process single time point
        storm_groups[storm_id].append({
            # Position data
            'position': {
                'lat': record['LAT'],
                'lon': record['LON']
            },
            # Wind speeds are in knots, convert to mph
            'wind_speed': record['USA_WIND'] * 1.15078 if record['USA_WIND'] is not None else None,
            # Pressure in millibars/hPa
            'pressure': record['USA_PRES'],
            # Storm status (may need normalisation)
            'status': record.get('USA_STATUS') or record.get('NATURE'),
            # Record time
            'timestamp': timestamp,
            # Radius of maximum winds (if available)
            'rmw': record.get('USA_RMW'),
            # Storm size - radius of 34kt winds (if available)
            'r34': record.get('USA_R34'),
            # Basin
            'basin': record.get('BASIN')
        })
    
    # Convert storm groups to processed hurricanes
    processed_hurricanes = []
    
    for storm_id, track in storm_groups.items():
        # Sort track by timestamp
        sorted_track = sorted(track, key=lambda x: x['timestamp'])
        
        # Get name from record
        # Find the first record with a name field
        record_with_name = next((r for r in records if r['SID'] == storm_id and r.get('NAME')), None)
        name = record_with_name['NAME'] if record_with_name else f'Unnamed Storm {storm_id}'
        
        # Get max wind speed to determine category
        max_wind_speed = max((p.get('wind_speed', 0) or 0) for p in sorted_track)
        category = get_hurricane_category(max_wind_speed)
        
        # Get year/season
        year = sorted_track[0]['timestamp'].year if sorted_track else pd.Timestamp.now().year
        
        # Add to processed hurricanes
        processed_hurricanes.append({
            'id': storm_id,
            'name': name,
            'year': year,
            'category': category,
            'basin': sorted_track[0].get('basin') if sorted_track else None,
            'start_time': sorted_track[0]['timestamp'] if sorted_track else None,
            'initial_position': sorted_track[0]['position'] if sorted_track else None,
            'initial_wind_speed': sorted_track[0]['wind_speed'] if sorted_track else None,
            'initial_pressure': sorted_track[0]['pressure'] if sorted_track else None,
            'track': sorted_track
        })
    
    # Ensure all hurricane IDs are unique by adding a suffix if needed
    unique_id_map = {}
    unique_hurricanes = []
    
    for hurricane in processed_hurricanes:
        unique_id = hurricane['id']
        
        # Check if I've seen this ID before
        if unique_id in unique_id_map:
            unique_id_map[unique_id] += 1
            unique_id = f"{hurricane['id']}-{unique_id_map[unique_id]}"
        else:
            unique_id_map[unique_id] = 1
        
        # Create a copy with the unique ID
        unique_hurricanes.append({
            **hurricane,
            'id': unique_id
        })
    
    return unique_hurricanes

def filter_hurricanes(hurricanes: List[Dict], options: Dict = None) -> List[Dict]:
    """
    Filters hurricanes based on criteria.
    
    Args:
        hurricanes: Processed hurricane data
        options: Filter options
        
    Returns:
        Filtered hurricane data
    """
    if options is None:
        options = {}
        
    filtered = hurricanes.copy()
    
    # Filter by basin
    if 'basin' in options:
        filtered = [h for h in filtered if h.get('basin') == options['basin']]
    
    # Filter by year/season
    if 'year' in options:
        filtered = [h for h in filtered if h.get('year') == options['year']]
    
    # Filter by minimum category
    if 'min_category' in options:
        min_cat = options['min_category']
        filtered = [h for h in filtered if category_to_numeric(h.get('category')) >= min_cat]
    
    # Filter to ensure adequate track data
    if 'min_track_points' in options:
        min_points = options['min_track_points']
        filtered = [h for h in filtered if len(h.get('track', [])) >= min_points]
    
    return filtered

def category_to_numeric(category: Union[str, int]) -> int:
    """
    Convert category to numeric value.
    
    Args:
        category: Category as string or number
        
    Returns:
        Numeric category value
    """
    if category is None:
        return 0
        
    if isinstance(category, int):
        return category
        
    if category == 'TD':
        return 0
    elif category == 'TS':
        return 0
    else:
        try:
            return int(category)
        except (ValueError, TypeError):
            return 0

async def fetch_ibtracs_hurricanes(options: Dict = None) -> List[Dict]:
    """
    Main function to fetch and process IBTrACS data.
    
    Args:
        options: Options for data fetching and filtering
        
    Returns:
        Processed hurricane data
    """
    if options is None:
        options = {}
        
    try:
        # Default to recent data for faster loading during development
        dataset = 'since1980' if options.get('full_history') else 'last3years'
        
        # Fetch raw data
        raw_data = await fetch_ibtracs_data(dataset)
        
        # Parse CSV data
        records = await parse_ibtracs_data(raw_data)
        
        # Process records into hurricane objects
        hurricanes = process_ibtracs_records(records)
        
        # Apply filters
        return filter_hurricanes(hurricanes, options)
    except Exception as e:
        logger.error(f'Error in fetch_ibtracs_hurricanes: {e}')
        raise e

def get_basin_stats(hurricanes: List[Dict]) -> Dict:
    """
    Utility function to get basins from processed hurricanes.
    
    Args:
        hurricanes: Processed hurricane data
        
    Returns:
        Object with basin codes and counts
    """
    basin_counts = {}
    basin_names = {
        'NA': 'North Atlantic',
        'SA': 'South Atlantic',
        'EP': 'Eastern Pacific',
        'WP': 'Western Pacific',
        'NI': 'North Indian',
        'SI': 'South Indian',
        'SP': 'South Pacific',
        'MM': 'Multi-basin'  # For storms that cross between basins
    }
    
    for h in hurricanes:
        basin = h.get('basin') or 'UNKNOWN'
        basin_counts[basin] = basin_counts.get(basin, 0) + 1
    
    return {
        'counts': basin_counts,
        'names': basin_names
    }