"""
evaluate.py - Evaluation script for hurricane prediction agent

Usage: python evaluate.py

This script tests the hurricane prediction agent against historical hurricane data
and reports performance metrics for inclusion in academic reports.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to import the hurricane_agents module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent and utility functions
from hurricane_agents.agent import HurricanePredictionAgent
from hurricane_agents.utils import haversine_distance, get_hurricane_category

# Test hurricanes from 2020-2024 (not in training data)
TEST_HURRICANES = [
    {
        "id": "test_ida_2021",
        "name": "Hurricane Ida",
        "year": 2021,
        "basin": "NA",  # North Atlantic
        "initial_position": {"lat": 25.1, "lon": -80.5},
        "initial_wind_speed": 75,
        "initial_pressure": 985,
        "track": [
            # Format: time, lat, lon, wind_speed, pressure
            {"time": 0, "position": {"lat": 25.1, "lon": -80.5}, "wind_speed": 75, "pressure": 985},
            {"time": 24, "position": {"lat": 26.8, "lon": -85.3}, "wind_speed": 100, "pressure": 968},
            {"time": 48, "position": {"lat": 28.2, "lon": -89.6}, "wind_speed": 140, "pressure": 935},
            {"time": 72, "position": {"lat": 30.5, "lon": -91.4}, "wind_speed": 120, "pressure": 945},
            {"time": 96, "position": {"lat": 34.7, "lon": -87.8}, "wind_speed": 60, "pressure": 990},
            {"time": 120, "position": {"lat": 39.2, "lon": -80.3}, "wind_speed": 35, "pressure": 1005}
        ]
    },
    {
        "id": "test_fiona_2022",
        "name": "Hurricane Fiona",
        "year": 2022,
        "basin": "NA",
        "initial_position": {"lat": 16.8, "lon": -62.1},
        "initial_wind_speed": 85,
        "initial_pressure": 980,
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
        "id": "test_dorian_2019",
        "name": "Hurricane Dorian",
        "year": 2019,
        "basin": "NA",
        "initial_position": {"lat": 18.3, "lon": -65.0},
        "initial_wind_speed": 80,
        "initial_pressure": 983,
        "track": [
            {"time": 0, "position": {"lat": 18.3, "lon": -65.0}, "wind_speed": 80, "pressure": 983},
            {"time": 24, "position": {"lat": 20.1, "lon": -67.8}, "wind_speed": 105, "pressure": 972},
            {"time": 48, "position": {"lat": 22.5, "lon": -70.3}, "wind_speed": 130, "pressure": 950},
            {"time": 72, "position": {"lat": 25.4, "lon": -73.1}, "wind_speed": 150, "pressure": 930},
            {"time": 96, "position": {"lat": 26.8, "lon": -76.7}, "wind_speed": 180, "pressure": 910},
            {"time": 120, "position": {"lat": 27.1, "lon": -78.4}, "wind_speed": 175, "pressure": 915}
        ]
    },
    {
        "id": "test_hagibis_2019",
        "name": "Typhoon Hagibis",
        "year": 2019,
        "basin": "WP",  # Western Pacific
        "initial_position": {"lat": 15.3, "lon": 153.2},
        "initial_wind_speed": 90,
        "initial_pressure": 975,
        "track": [
            {"time": 0, "position": {"lat": 15.3, "lon": 153.2}, "wind_speed": 90, "pressure": 975},
            {"time": 24, "position": {"lat": 16.7, "lon": 150.1}, "wind_speed": 125, "pressure": 950},
            {"time": 48, "position": {"lat": 19.2, "lon": 146.3}, "wind_speed": 155, "pressure": 915},
            {"time": 72, "position": {"lat": 23.7, "lon": 141.2}, "wind_speed": 140, "pressure": 925},
            {"time": 96, "position": {"lat": 29.4, "lon": 138.6}, "wind_speed": 120, "pressure": 945},
            {"time": 120, "position": {"lat": 36.2, "lon": 142.1}, "wind_speed": 85, "pressure": 975}
        ]
    },
    {
        "id": "test_elsa_2021",
        "name": "Hurricane Elsa",
        "year": 2021,
        "basin": "NA",
        "initial_position": {"lat": 11.2, "lon": -52.8},
        "initial_wind_speed": 60,
        "initial_pressure": 1000,
        "track": [
            {"time": 0, "position": {"lat": 11.2, "lon": -52.8}, "wind_speed": 60, "pressure": 1000},
            {"time": 24, "position": {"lat": 13.1, "lon": -57.6}, "wind_speed": 75, "pressure": 995},
            {"time": 48, "position": {"lat": 15.3, "lon": -63.7}, "wind_speed": 80, "pressure": 991},
            {"time": 72, "position": {"lat": 18.6, "lon": -70.5}, "wind_speed": 65, "pressure": 1002},
            {"time": 96, "position": {"lat": 22.9, "lon": -79.8}, "wind_speed": 60, "pressure": 1005},
            {"time": 120, "position": {"lat": 28.7, "lon": -82.3}, "wind_speed": 55, "pressure": 1007}
        ]
    }
]

def evaluate_agent(agent: HurricanePredictionAgent) -> Dict[str, Any]:
    """
    Evaluate the hurricane prediction agent against test hurricanes.
    
    Args:
        agent: The hurricane prediction agent
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        "overall": {
            "position_errors": [],
            "wind_speed_errors": [],
            "pressure_errors": [],
            "position_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []},
            "wind_speed_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []},
            "pressure_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []}
        },
        "by_hurricane": {},
        "by_basin": {}
    }
    
    # Process each test hurricane
    for hurricane in TEST_HURRICANES:
        hurricane_id = hurricane["id"]
        basin = hurricane["basin"]
        
        # Initialize results for this hurricane and basin if needed
        if hurricane_id not in results["by_hurricane"]:
            results["by_hurricane"][hurricane_id] = {
                "position_errors": [],
                "wind_speed_errors": [],
                "pressure_errors": [],
                "position_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []},
                "wind_speed_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []},
                "pressure_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []}
            }
        
        if basin not in results["by_basin"]:
            results["by_basin"][basin] = {
                "position_errors": [],
                "wind_speed_errors": [],
                "pressure_errors": [],
                "position_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []},
                "wind_speed_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []},
                "pressure_errors_by_time": {24: [], 48: [], 72: [], 96: [], 120: []}
            }
        
        # Store initial state and history
        initial_state = {
            "position": hurricane["initial_position"],
            "wind_speed": hurricane["initial_wind_speed"],
            "pressure": hurricane["initial_pressure"],
            "basin": hurricane["basin"],
            "sea_surface_temp": {"value": 28.5} # Default value
        }
        
        history = []
        
        # Generate predictions and compare with actual track
        for track_point in hurricane["track"][1:]:  # Skip the first point (initial state)
            time_hours = track_point["time"]
            
            # Make prediction
            prediction = agent.predict(initial_state, history, training=False)
            
            # Calculate errors
            position_error = haversine_distance(
                prediction.get("position", {"lat": 0, "lon": 0}),
                track_point["position"]
            )
            
            wind_speed_error = abs(prediction.get("wind_speed", 0) - track_point["wind_speed"])
            pressure_error = abs(prediction.get("pressure", 1010) - track_point["pressure"])
            
            # Store errors in results
            results["overall"]["position_errors"].append(position_error)
            results["overall"]["wind_speed_errors"].append(wind_speed_error)
            results["overall"]["pressure_errors"].append(pressure_error)
            
            results["by_hurricane"][hurricane_id]["position_errors"].append(position_error)
            results["by_hurricane"][hurricane_id]["wind_speed_errors"].append(wind_speed_error)
            results["by_hurricane"][hurricane_id]["pressure_errors"].append(pressure_error)
            
            results["by_basin"][basin]["position_errors"].append(position_error)
            results["by_basin"][basin]["wind_speed_errors"].append(wind_speed_error)
            results["by_basin"][basin]["pressure_errors"].append(pressure_error)
            
            # Store by forecast time
            if time_hours in results["overall"]["position_errors_by_time"]:
                results["overall"]["position_errors_by_time"][time_hours].append(position_error)
                results["overall"]["wind_speed_errors_by_time"][time_hours].append(wind_speed_error)
                results["overall"]["pressure_errors_by_time"][time_hours].append(pressure_error)
                
                results["by_hurricane"][hurricane_id]["position_errors_by_time"][time_hours].append(position_error)
                results["by_hurricane"][hurricane_id]["wind_speed_errors_by_time"][time_hours].append(wind_speed_error)
                results["by_hurricane"][hurricane_id]["pressure_errors_by_time"][time_hours].append(pressure_error)
                
                results["by_basin"][basin]["position_errors_by_time"][time_hours].append(position_error)
                results["by_basin"][basin]["wind_speed_errors_by_time"][time_hours].append(wind_speed_error)
                results["by_basin"][basin]["pressure_errors_by_time"][time_hours].append(pressure_error)
    
    # Calculate summary statistics
    # Overall
    results["summary"] = {
        "overall": {
            "avg_position_error": np.mean(results["overall"]["position_errors"]),
            "avg_wind_speed_error": np.mean(results["overall"]["wind_speed_errors"]),
            "avg_pressure_error": np.mean(results["overall"]["pressure_errors"]),
            "by_time": {}
        },
        "by_hurricane": {},
        "by_basin": {}
    }
    
    # By time
    for time_hours in [24, 48, 72, 96, 120]:
        position_errors = results["overall"]["position_errors_by_time"][time_hours]
        wind_speed_errors = results["overall"]["wind_speed_errors_by_time"][time_hours]
        pressure_errors = results["overall"]["pressure_errors_by_time"][time_hours]
        
        if position_errors:
            results["summary"]["overall"]["by_time"][time_hours] = {
                "avg_position_error": np.mean(position_errors),
                "avg_wind_speed_error": np.mean(wind_speed_errors),
                "avg_pressure_error": np.mean(pressure_errors)
            }
    
    # By hurricane
    for hurricane_id in results["by_hurricane"]:
        results["summary"]["by_hurricane"][hurricane_id] = {
            "avg_position_error": np.mean(results["by_hurricane"][hurricane_id]["position_errors"]),
            "avg_wind_speed_error": np.mean(results["by_hurricane"][hurricane_id]["wind_speed_errors"]),
            "avg_pressure_error": np.mean(results["by_hurricane"][hurricane_id]["pressure_errors"]),
            "by_time": {}
        }
        
        for time_hours in [24, 48, 72, 96, 120]:
            position_errors = results["by_hurricane"][hurricane_id]["position_errors_by_time"][time_hours]
            wind_speed_errors = results["by_hurricane"][hurricane_id]["wind_speed_errors_by_time"][time_hours]
            pressure_errors = results["by_hurricane"][hurricane_id]["pressure_errors_by_time"][time_hours]
            
            if position_errors:
                results["summary"]["by_hurricane"][hurricane_id]["by_time"][time_hours] = {
                    "avg_position_error": np.mean(position_errors),
                    "avg_wind_speed_error": np.mean(wind_speed_errors),
                    "avg_pressure_error": np.mean(pressure_errors)
                }
    
    # By basin
    for basin in results["by_basin"]:
        results["summary"]["by_basin"][basin] = {
            "avg_position_error": np.mean(results["by_basin"][basin]["position_errors"]),
            "avg_wind_speed_error": np.mean(results["by_basin"][basin]["wind_speed_errors"]),
            "avg_pressure_error": np.mean(results["by_basin"][basin]["pressure_errors"]),
            "by_time": {}
        }
        
        for time_hours in [24, 48, 72, 96, 120]:
            position_errors = results["by_basin"][basin]["position_errors_by_time"][time_hours]
            wind_speed_errors = results["by_basin"][basin]["wind_speed_errors_by_time"][time_hours]
            pressure_errors = results["by_basin"][basin]["pressure_errors_by_time"][time_hours]
            
            if position_errors:
                results["summary"]["by_basin"][basin]["by_time"][time_hours] = {
                    "avg_position_error": np.mean(position_errors),
                    "avg_wind_speed_error": np.mean(wind_speed_errors),
                    "avg_pressure_error": np.mean(pressure_errors)
                }
    
    return results

def plot_results(results: Dict[str, Any], output_dir: str = "evaluation_results"):
    """
    Plot evaluation results.
    
    Args:
        results: Dictionary with evaluation metrics
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Position error by forecast time
    plt.figure(figsize=(10, 6))
    times = [24, 48, 72, 96, 120]
    avg_errors = [results["summary"]["overall"]["by_time"][t]["avg_position_error"] for t in times]
    plt.plot(times, avg_errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Forecast Time (hours)')
    plt.ylabel('Average Position Error (km)')
    plt.title('Hurricane Position Error by Forecast Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'position_error_by_time.png'))
    
    # Plot 2: Wind speed error by forecast time
    plt.figure(figsize=(10, 6))
    avg_errors = [results["summary"]["overall"]["by_time"][t]["avg_wind_speed_error"] for t in times]
    plt.plot(times, avg_errors, 'o-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Forecast Time (hours)')
    plt.ylabel('Average Wind Speed Error (mph)')
    plt.title('Hurricane Intensity Error by Forecast Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'wind_speed_error_by_time.png'))
    
    # Plot 3: Position error by hurricane
    plt.figure(figsize=(12, 6))
    hurricane_names = [next(h["name"] for h in TEST_HURRICANES if h["id"] == hid) 
                     for hid in results["summary"]["by_hurricane"].keys()]
    avg_errors = [results["summary"]["by_hurricane"][hid]["avg_position_error"] 
                for hid in results["summary"]["by_hurricane"].keys()]
    
    plt.bar(hurricane_names, avg_errors, color='skyblue')
    plt.xlabel('Hurricane')
    plt.ylabel('Average Position Error (km)')
    plt.title('Average Position Error by Hurricane')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_error_by_hurricane.png'))
    
    # Plot 4: Position error by basin
    plt.figure(figsize=(10, 6))
    basin_names = list(results["summary"]["by_basin"].keys())
    basin_full_names = {'NA': 'North Atlantic', 'EP': 'Eastern Pacific', 
                        'WP': 'Western Pacific', 'NI': 'North Indian', 
                        'SI': 'South Indian', 'SP': 'South Pacific'}
    
    basin_labels = [basin_full_names.get(b, b) for b in basin_names]
    avg_errors = [results["summary"]["by_basin"][b]["avg_position_error"] for b in basin_names]
    
    plt.bar(basin_labels, avg_errors, color='lightgreen')
    plt.xlabel('Ocean Basin')
    plt.ylabel('Average Position Error (km)')
    plt.title('Average Position Error by Ocean Basin')
    plt.savefig(os.path.join(output_dir, 'position_error_by_basin.png'))
    
    # Plot 5: Combined plot for published paper
    plt.figure(figsize=(12, 10))
    
    # First subplot: Position error by time
    plt.subplot(2, 2, 1)
    times = [24, 48, 72, 96, 120]
    avg_errors = [results["summary"]["overall"]["by_time"][t]["avg_position_error"] for t in times]
    plt.plot(times, avg_errors, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Forecast Time (hours)')
    plt.ylabel('Avg Position Error (km)')
    plt.title('Position Error by Forecast Time')
    plt.grid(True)
    
    # Second subplot: Intensity error by time
    plt.subplot(2, 2, 2)
    avg_errors = [results["summary"]["overall"]["by_time"][t]["avg_wind_speed_error"] for t in times]
    plt.plot(times, avg_errors, 'o-', linewidth=2, markersize=8, color='red')
    plt.xlabel('Forecast Time (hours)')
    plt.ylabel('Avg Wind Speed Error (mph)')
    plt.title('Intensity Error by Forecast Time')
    plt.grid(True)
    
    # Third subplot: Error by hurricane
    plt.subplot(2, 2, 3)
    hurricane_names = [next(h["name"] for h in TEST_HURRICANES if h["id"] == hid).split(' ')[1]
                     for hid in results["summary"]["by_hurricane"].keys()]
    pos_errors = [results["summary"]["by_hurricane"][hid]["avg_position_error"] 
                for hid in results["summary"]["by_hurricane"].keys()]
    int_errors = [results["summary"]["by_hurricane"][hid]["avg_wind_speed_error"] 
                for hid in results["summary"]["by_hurricane"].keys()]
    
    x = np.arange(len(hurricane_names))
    width = 0.35
    plt.bar(x - width/2, pos_errors, width, label='Position (km)')
    plt.bar(x + width/2, int_errors, width, label='Intensity (mph)')
    plt.xlabel('Hurricane')
    plt.ylabel('Average Error')
    plt.title('Error by Hurricane')
    plt.xticks(x, hurricane_names)
    plt.legend()
    
    # Fourth subplot: Error by basin
    plt.subplot(2, 2, 4)
    basin_errors = {}
    for basin in results["summary"]["by_basin"]:
        for time in [24, 48, 72]:
            if time in results["summary"]["by_basin"][basin]["by_time"]:
                if basin not in basin_errors:
                    basin_errors[basin] = []
                basin_errors[basin].append(results["summary"]["by_basin"][basin]["by_time"][time]["avg_position_error"])
    
    basin_labels = [basin_full_names.get(b, b) for b in basin_errors.keys()]
    basin_avg_errors = [np.mean(errors) for errors in basin_errors.values()]
    
    plt.bar(basin_labels, basin_avg_errors, color='purple')
    plt.xlabel('Ocean Basin')
    plt.ylabel('Avg Position Error (km)')
    plt.title('Error by Ocean Basin (24-72h)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_performance_metrics.png'))
    
    print(f"Plots saved to {output_dir} directory")

def print_results(results: Dict[str, Any]):
    """
    Print evaluation results to console.
    
    Args:
        results: Dictionary with evaluation metrics
    """
    print("\n===== HURRICANE PREDICTION AGENT EVALUATION =====\n")
    
    # Overall performance
    print("OVERALL PERFORMANCE:")
    print(f"  Average Position Error: {results['summary']['overall']['avg_position_error']:.2f} km")
    print(f"  Average Wind Speed Error: {results['summary']['overall']['avg_wind_speed_error']:.2f} mph")
    print(f"  Average Pressure Error: {results['summary']['overall']['avg_pressure_error']:.2f} mb")
    
    # Performance by forecast time
    print("\nPERFORMANCE BY FORECAST TIME:")
    for time in sorted(results["summary"]["overall"]["by_time"].keys()):
        time_results = results["summary"]["overall"]["by_time"][time]
        print(f"  {time} hours:")
        print(f"    Position Error: {time_results['avg_position_error']:.2f} km")
        print(f"    Wind Speed Error: {time_results['avg_wind_speed_error']:.2f} mph")
        print(f"    Pressure Error: {time_results['avg_pressure_error']:.2f} mb")
    
    # Performance by basin
    print("\nPERFORMANCE BY BASIN:")
    basin_full_names = {'NA': 'North Atlantic', 'EP': 'Eastern Pacific', 
                        'WP': 'Western Pacific', 'NI': 'North Indian', 
                        'SI': 'South Indian', 'SP': 'South Pacific'}
    
    for basin in sorted(results["summary"]["by_basin"].keys()):
        basin_name = basin_full_names.get(basin, basin)
        basin_results = results["summary"]["by_basin"][basin]
        print(f"  {basin_name}:")
        print(f"    Position Error: {basin_results['avg_position_error']:.2f} km")
        print(f"    Wind Speed Error: {basin_results['avg_wind_speed_error']:.2f} mph")
        print(f"    Pressure Error: {basin_results['avg_pressure_error']:.2f} mb")
    
    # Performance by hurricane
    print("\nPERFORMANCE BY HURRICANE:")
    for hurricane_id in sorted(results["summary"]["by_hurricane"].keys()):
        hurricane_name = next(h["name"] for h in TEST_HURRICANES if h["id"] == hurricane_id)
        hurricane_results = results["summary"]["by_hurricane"][hurricane_id]
        print(f"  {hurricane_name}:")
        print(f"    Position Error: {hurricane_results['avg_position_error']:.2f} km")
        print(f"    Wind Speed Error: {hurricane_results['avg_wind_speed_error']:.2f} mph")
        print(f"    Pressure Error: {hurricane_results['avg_pressure_error']:.2f} mb")
    
    print("\n==========================================")

def save_results(results: Dict[str, Any], output_file: str = "evaluation_results/results.json"):
    """
    Save evaluation results to file.
    
    Args:
        results: Dictionary with evaluation metrics
        output_file: File to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    """Main function to evaluate the hurricane prediction agent."""
    print("Initializing hurricane prediction agent...")
    
    # Create agent with default configuration
    agent = HurricanePredictionAgent()
    
    # Run evaluation
    print("Evaluating agent on test hurricanes...")
    results = evaluate_agent(agent)
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results)
    
    # Plot results
    plot_results(results)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()