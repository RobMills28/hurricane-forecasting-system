"""
evaluate.py - Evaluation script for hurricane and storm prediction agent
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent and utility functions
from hurricane_agents.agent import HurricanePredictionAgent
from hurricane_agents.utils import haversine_distance, get_hurricane_category

# Test hurricanes and other storm types
TEST_STORMS = [
    # Original hurricanes
    {
        "id": "test_ida_2021", "name": "Hurricane Ida", "year": 2021, "basin": "NA", "type": "Hurricane",
        "initial_position": {"lat": 25.1, "lon": -80.5}, "initial_wind_speed": 75, "initial_pressure": 985,
        "track": [
            {"time": 0, "position": {"lat": 25.1, "lon": -80.5}, "wind_speed": 75, "pressure": 985},
            {"time": 24, "position": {"lat": 26.8, "lon": -85.3}, "wind_speed": 100, "pressure": 968},
            {"time": 48, "position": {"lat": 28.2, "lon": -89.6}, "wind_speed": 140, "pressure": 935},
            {"time": 72, "position": {"lat": 30.5, "lon": -91.4}, "wind_speed": 120, "pressure": 945},
            {"time": 96, "position": {"lat": 34.7, "lon": -87.8}, "wind_speed": 60, "pressure": 990},
            {"time": 120, "position": {"lat": 39.2, "lon": -80.3}, "wind_speed": 35, "pressure": 1005}
        ]
    },
    {
        "id": "test_fiona_2022", "name": "Hurricane Fiona", "year": 2022, "basin": "NA", "type": "Hurricane",
        "initial_position": {"lat": 16.8, "lon": -62.1}, "initial_wind_speed": 85, "initial_pressure": 980,
        "track": [
            {"time": 0, "position": {"lat": 16.8, "lon": -62.1}, "wind_speed": 85, "pressure": 980},
            {"time": 24, "position": {"lat": 18.2, "lon": -65.8}, "wind_speed": 95, "pressure": 975},
            {"time": 48, "position": {"lat": 19.7, "lon": -69.2}, "wind_speed": 110, "pressure": 960},
            {"time": 72, "position": {"lat": 22.6, "lon": -71.5}, "wind_speed": 125, "pressure": 945},
            {"time": 96, "position": {"lat": 27.3, "lon": -68.4}, "wind_speed": 130, "pressure": 940},
            {"time": 120, "position": {"lat": 35.6, "lon": -58.9}, "wind_speed": 110, "pressure": 955}
        ]
    },
    {
        "id": "test_hagibis_2019", "name": "Typhoon Hagibis", "year": 2019, "basin": "WP", "type": "Typhoon",
        "initial_position": {"lat": 15.3, "lon": 153.2}, "initial_wind_speed": 90, "initial_pressure": 975,
        "track": [
            {"time": 0, "position": {"lat": 15.3, "lon": 153.2}, "wind_speed": 90, "pressure": 975},
            {"time": 24, "position": {"lat": 16.7, "lon": 150.1}, "wind_speed": 125, "pressure": 950},
            {"time": 48, "position": {"lat": 19.2, "lon": 146.3}, "wind_speed": 155, "pressure": 915},
            {"time": 72, "position": {"lat": 23.7, "lon": 141.2}, "wind_speed": 140, "pressure": 925},
            {"time": 96, "position": {"lat": 29.4, "lon": 138.6}, "wind_speed": 120, "pressure": 945},
            {"time": 120, "position": {"lat": 36.2, "lon": 142.1}, "wind_speed": 85, "pressure": 975}
        ]
    },
    # Additional storm types
    {
        "id": "test_winter_storm_2023", "name": "Winter Storm Elliott", "year": 2023, "basin": "NA", "type": "Winter Storm",
        "initial_position": {"lat": 42.5, "lon": -95.3}, "initial_wind_speed": 45, "initial_pressure": 992,
        "track": [
            {"time": 0, "position": {"lat": 42.5, "lon": -95.3}, "wind_speed": 45, "pressure": 992},
            {"time": 24, "position": {"lat": 43.8, "lon": -89.2}, "wind_speed": 50, "pressure": 988},
            {"time": 48, "position": {"lat": 45.1, "lon": -84.7}, "wind_speed": 55, "pressure": 982},
            {"time": 72, "position": {"lat": 46.3, "lon": -78.5}, "wind_speed": 48, "pressure": 985},
            {"time": 96, "position": {"lat": 47.2, "lon": -72.1}, "wind_speed": 42, "pressure": 990},
            {"time": 120, "position": {"lat": 48.7, "lon": -65.3}, "wind_speed": 35, "pressure": 995}
        ]
    },
    {
        "id": "test_thunderstorm_2023", "name": "Severe Thunderstorm Texas", "year": 2023, "basin": "NA", "type": "Severe Thunderstorm",
        "initial_position": {"lat": 32.1, "lon": -96.8}, "initial_wind_speed": 65, "initial_pressure": 998,
        "track": [
            {"time": 0, "position": {"lat": 32.1, "lon": -96.8}, "wind_speed": 65, "pressure": 998},
            {"time": 24, "position": {"lat": 33.2, "lon": -94.3}, "wind_speed": 70, "pressure": 995},
            {"time": 48, "position": {"lat": 34.5, "lon": -91.8}, "wind_speed": 60, "pressure": 997},
            {"time": 72, "position": {"lat": 35.7, "lon": -89.2}, "wind_speed": 50, "pressure": 1000},
            {"time": 96, "position": {"lat": 36.9, "lon": -87.1}, "wind_speed": 35, "pressure": 1004},
            {"time": 120, "position": {"lat": 38.2, "lon": -84.5}, "wind_speed": 25, "pressure": 1008}
        ]
    }
]

def evaluate_agent(agent: HurricanePredictionAgent) -> Dict[str, Any]:
    """Evaluate the agent against test storms."""
    results = {
        "overall": {
            "position_errors": [], "wind_speed_errors": [], "pressure_errors": [],
            "position_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []},
            "wind_speed_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []},
            "pressure_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []}
        },
        "by_storm": {}, "by_basin": {}, "by_type": {}
    }
    
    # Process each test storm
    for storm in TEST_STORMS:
        storm_id, basin, storm_type = storm["id"], storm["basin"], storm["type"]
        
        # Initialize results containers if needed
        for container_type, key in [("by_storm", storm_id), ("by_basin", basin), ("by_type", storm_type)]:
            if key not in results[container_type]:
                results[container_type][key] = {
                    "position_errors": [], "wind_speed_errors": [], "pressure_errors": [],
                    "position_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []},
                    "wind_speed_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []}
                }
        
        # Set up initial state and history
        initial_state = {
            "position": storm["initial_position"],
            "wind_speed": storm["initial_wind_speed"],
            "pressure": storm["initial_pressure"],
            "basin": storm["basin"],
            "sea_surface_temp": {"value": 28.5}
        }
        history = []
        
        # Generate predictions and compare with actual track
        for track_point in storm["track"][1:]:
            time_hours = track_point["time"]
            prediction = agent.predict(initial_state, history, training=False)
            
            # Calculate errors
            position_error = haversine_distance(
                prediction.get("position", {"lat": 0, "lon": 0}),
                track_point["position"]
            )
            wind_speed_error = abs(prediction.get("wind_speed", 0) - track_point["wind_speed"])
            pressure_error = abs(prediction.get("pressure", 1010) - track_point["pressure"])
            
            # Store errors in results
            for container_type, key in [("overall", None), ("by_storm", storm_id), 
                                       ("by_basin", basin), ("by_type", storm_type)]:
                container = results[container_type] if key is None else results[container_type][key]
                container["position_errors"].append(position_error)
                container["wind_speed_errors"].append(wind_speed_error)
                container["pressure_errors"].append(pressure_error)
                
                if time_hours in container["position_errors_by_time"]:
                    container["position_errors_by_time"][time_hours].append(position_error)
                    container["wind_speed_errors_by_time"][time_hours].append(wind_speed_error)
    
    # Calculate summary statistics
    results["summary"] = {
        "overall": {
            "avg_position_error": np.mean(results["overall"]["position_errors"]),
            "avg_wind_speed_error": np.mean(results["overall"]["wind_speed_errors"]),
            "avg_pressure_error": np.mean(results["overall"]["pressure_errors"]),
            "by_time": {}
        },
        "by_storm": {}, "by_basin": {}, "by_type": {}
    }
    
    # By time
    for time_hours in [24, 48, 72, 96, 120]:
        if results["overall"]["position_errors_by_time"][time_hours]:
            results["summary"]["overall"]["by_time"][time_hours] = {
                "avg_position_error": np.mean(results["overall"]["position_errors_by_time"][time_hours]),
                "avg_wind_speed_error": np.mean(results["overall"]["wind_speed_errors_by_time"][time_hours])
            }
    
    # By storm, basin, and type
    for container_type in ["by_storm", "by_basin", "by_type"]:
        for key in results[container_type]:
            if not results[container_type][key]["position_errors"]:
                continue
                
            results["summary"][container_type][key] = {
                "avg_position_error": np.mean(results[container_type][key]["position_errors"]),
                "avg_wind_speed_error": np.mean(results[container_type][key]["wind_speed_errors"]),
                "avg_pressure_error": np.mean(results[container_type][key]["pressure_errors"])
            }
    
    return results

def plot_results(results: Dict[str, Any], output_dir: str = "evaluation_results"):
    """Plot evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Position error by forecast time
    plt.figure(figsize=(10, 6))
    times = [24, 48, 72, 96, 120]
    avg_errors = [results["summary"]["overall"]["by_time"][t]["avg_position_error"] for t in times]
    plt.plot(times, avg_errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Forecast Time (hours)')
    plt.ylabel('Average Position Error (km)')
    plt.title('Storm Position Error by Forecast Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'position_error_by_time.png'))
    
    # Plot 2: Wind speed error by forecast time
    plt.figure(figsize=(10, 6))
    avg_errors = [results["summary"]["overall"]["by_time"][t]["avg_wind_speed_error"] for t in times]
    plt.plot(times, avg_errors, 'o-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Forecast Time (hours)')
    plt.ylabel('Average Wind Speed Error (mph)')
    plt.title('Storm Intensity Error by Forecast Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'wind_speed_error_by_time.png'))
    
    # Plot 3: Error by storm type (NEW)
    plt.figure(figsize=(12, 6))
    type_names = list(results["summary"]["by_type"].keys())
    type_errors = [results["summary"]["by_type"][t]["avg_position_error"] for t in type_names]
    type_intensity_errors = [results["summary"]["by_type"][t]["avg_wind_speed_error"] for t in type_names]
    
    x = np.arange(len(type_names))
    width = 0.35
    plt.bar(x - width/2, type_errors, width, label='Position (km)', color='royalblue')
    plt.bar(x + width/2, type_intensity_errors, width, label='Intensity (mph)', color='tomato')
    plt.xlabel('Storm Type')
    plt.ylabel('Average Error')
    plt.title('Error by Storm Type')
    plt.xticks(x, type_names, rotation=30, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_by_storm_type.png'))
    
    # Save results to file
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def print_results(results: Dict[str, Any]):
    """Print evaluation results to console."""
    print("\n===== STORM PREDICTION AGENT EVALUATION =====\n")
    
    # Overall performance
    print("OVERALL PERFORMANCE:")
    print(f"  Average Position Error: {results['summary']['overall']['avg_position_error']:.2f} km")
    print(f"  Average Wind Speed Error: {results['summary']['overall']['avg_wind_speed_error']:.2f} mph")
    
    # Performance by forecast time
    print("\nPERFORMANCE BY FORECAST TIME:")
    for time in sorted(results["summary"]["overall"]["by_time"].keys()):
        time_results = results["summary"]["overall"]["by_time"][time]
        print(f"  {time} hours:")
        print(f"    Position Error: {time_results['avg_position_error']:.2f} km")
        print(f"    Wind Speed Error: {time_results['avg_wind_speed_error']:.2f} mph")
    
    # Performance by storm type (NEW)
    print("\nPERFORMANCE BY STORM TYPE:")
    for storm_type in sorted(results["summary"]["by_type"].keys()):
        type_results = results["summary"]["by_type"][storm_type]
        print(f"  {storm_type}:")
        print(f"    Position Error: {type_results['avg_position_error']:.2f} km")
        print(f"    Wind Speed Error: {type_results['avg_wind_speed_error']:.2f} mph")
    
    print("\n==========================================")

def main():
    """Main function to evaluate the storm prediction agent."""
    print("Initializing storm prediction agent...")
    
    # Create agent with default configuration
    agent = HurricanePredictionAgent()
    
    # Run evaluation
    print("Evaluating agent on test storms...")
    results = evaluate_agent(agent)
    
    # Print results
    print_results(results)
    
    # Plot results
    plot_results(results)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()