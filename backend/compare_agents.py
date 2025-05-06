"""
Compare Agent Systems - Hurricane Prediction

This script evaluates and compares the single-agent and multi-agent approaches
for hurricane trajectory and intensity prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import both agent implementations
from hurricane_agents.single_agent import HurricanePredictionAgent
from hurricane_agents.ensemble import EnsembleCoordinator
from evaluate import evaluate_agent

def compare_agent_performance():
    """Compare performance of single agent vs multi-agent ensemble system."""
    print("\n===== HURRICANE AGENT COMPARISON =====\n")
    
    # Initialize both agent systems
    print("Initializing single agent and multi-agent systems...")
    single_agent = HurricanePredictionAgent()
    multi_agent = EnsembleCoordinator()
    
    # Run evaluations
    print("\nEvaluating single agent performance...")
    single_results = evaluate_agent(single_agent)
    
    print("\nEvaluating multi-agent ensemble performance...")
    multi_results = evaluate_agent(multi_agent)
    
    # Generate comparison visualizations
    print("\nGenerating comparison charts...")
    generate_comparison_charts(single_results, multi_results)
    
    # Print performance summary
    print("\n===== PERFORMANCE COMPARISON SUMMARY =====")
    print_comparison_summary(single_results, multi_results)
    
    return {
        "single_agent": single_results,
        "multi_agent": multi_results
    }

def generate_comparison_charts(single_results: Dict, multi_results: Dict):
    """Generate visualizations comparing both agent systems."""
    # Create output directory
    os.makedirs('comparison_results', exist_ok=True)
    
    # 1. Position Error by Forecast Time comparison
    plt.figure(figsize=(10, 6))
    times = [24, 48, 72, 96, 120]
    
    single_errors = [single_results["summary"]["overall"]["by_time"][t]["avg_position_error"] 
                    for t in times]
    multi_errors = [multi_results["summary"]["overall"]["by_time"][t]["avg_position_error"] 
                   for t in times]
    
    plt.plot(times, single_errors, 'o-', linewidth=2, label='Single Agent', color='blue')
    plt.plot(times, multi_errors, 'o-', linewidth=2, label='Multi-Agent Ensemble', color='green')
    plt.xlabel('Forecast Time (hours)')
    plt.ylabel('Average Position Error (km)')
    plt.title('Position Error Comparison by Forecast Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('comparison_results/position_error_comparison.png')
    
    # 2. Wind Speed Error by Forecast Time comparison
    plt.figure(figsize=(10, 6))
    
    single_wind_errors = [single_results["summary"]["overall"]["by_time"][t]["avg_wind_speed_error"] 
                         for t in times]
    multi_wind_errors = [multi_results["summary"]["overall"]["by_time"][t]["avg_wind_speed_error"] 
                        for t in times]
    
    plt.plot(times, single_wind_errors, 'o-', linewidth=2, label='Single Agent', color='blue')
    plt.plot(times, multi_wind_errors, 'o-', linewidth=2, label='Multi-Agent Ensemble', color='green')
    plt.xlabel('Forecast Time (hours)')
    plt.ylabel('Average Wind Speed Error (mph)')
    plt.title('Intensity Error Comparison by Forecast Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('comparison_results/intensity_error_comparison.png')
    
    # 3. Performance by Basin
    plt.figure(figsize=(12, 6))
    
    basins = list(single_results["summary"]["by_basin"].keys())
    basin_labels = []
    for basin in basins:
        if basin == 'NA':
            basin_labels.append('North Atlantic')
        elif basin == 'WP':
            basin_labels.append('Western Pacific')
        elif basin == 'EP':
            basin_labels.append('Eastern Pacific')
        else:
            basin_labels.append(basin)
    
    single_basin_errors = [single_results["summary"]["by_basin"][b]["avg_position_error"] 
                          for b in basins]
    multi_basin_errors = [multi_results["summary"]["by_basin"][b]["avg_position_error"] 
                         for b in basins]
    
    x = np.arange(len(basins))
    width = 0.35
    
    plt.bar(x - width/2, single_basin_errors, width, label='Single Agent', color='blue')
    plt.bar(x + width/2, multi_basin_errors, width, label='Multi-Agent Ensemble', color='green')
    plt.xlabel('Ocean Basin')
    plt.ylabel('Average Position Error (km)')
    plt.title('Position Error Comparison by Ocean Basin')
    plt.xticks(x, basin_labels)
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_results/basin_error_comparison.png')
    
    # 4. Performance by Hurricane (stacked bar chart)
    plt.figure(figsize=(14, 8))
    
    hurricanes = list(single_results["summary"]["by_storm"].keys())
    hurricane_names = []
    for hurricane in hurricanes:
        # Try to find the hurricane name in the summary data
        # If not found, just use the ID as the name
        hurricane_name = hurricane  # Default to using the ID
        
        # Look for the name in by_storm if it's a list of dictionaries
        if isinstance(single_results.get("by_storm"), list):
            for storm in single_results["by_storm"]:
                if isinstance(storm, dict) and storm.get("id") == hurricane:
                    hurricane_name = storm.get("name", hurricane)
                    break
        
        hurricane_names.append(hurricane_name)
    
    single_hurricane_errors = [single_results["summary"]["by_storm"][h]["avg_position_error"] 
                              for h in hurricanes]
    multi_hurricane_errors = [multi_results["summary"]["by_storm"][h]["avg_position_error"] 
                             for h in hurricanes]
    
    x = np.arange(len(hurricanes))
    width = 0.35
    
    plt.bar(x - width/2, single_hurricane_errors, width, label='Single Agent', color='blue')
    plt.bar(x + width/2, multi_hurricane_errors, width, label='Multi-Agent Ensemble', color='green')
    plt.xlabel('Hurricane')
    plt.ylabel('Average Position Error (km)')
    plt.title('Position Error Comparison by Hurricane')
    plt.xticks(x, hurricane_names, rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_results/hurricane_error_comparison.png')
    
    # 5. Overall comparison (side-by-side)
    plt.figure(figsize=(12, 6))
    
    metrics = ['Position Error (km)', 'Wind Speed Error (mph)', 'Pressure Error (mb)']
    single_overall = [
        single_results["summary"]["overall"]["avg_position_error"],
        single_results["summary"]["overall"]["avg_wind_speed_error"],
        single_results["summary"]["overall"]["avg_pressure_error"]
    ]
    multi_overall = [
        multi_results["summary"]["overall"]["avg_position_error"],
        multi_results["summary"]["overall"]["avg_wind_speed_error"],
        multi_results["summary"]["overall"]["avg_pressure_error"]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, single_overall, width, label='Single Agent', color='blue')
    plt.bar(x + width/2, multi_overall, width, label='Multi-Agent Ensemble', color='green')
    plt.xlabel('Metric')
    plt.ylabel('Error')
    plt.title('Overall Performance Comparison')
    plt.xticks(x, metrics)
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_results/overall_comparison.png')
    
    print(f"Comparison charts saved to comparison_results directory")

def print_comparison_summary(single_results: Dict, multi_results: Dict):
    """Print a summary of the performance comparison."""
    # Calculate improvement percentages
    pos_error_single = single_results["summary"]["overall"]["avg_position_error"]
    pos_error_multi = multi_results["summary"]["overall"]["avg_position_error"]
    pos_improvement = ((pos_error_single - pos_error_multi) / pos_error_single) * 100
    
    wind_error_single = single_results["summary"]["overall"]["avg_wind_speed_error"]
    wind_error_multi = multi_results["summary"]["overall"]["avg_wind_speed_error"]
    wind_improvement = ((wind_error_single - wind_error_multi) / wind_error_single) * 100
    
    pressure_error_single = single_results["summary"]["overall"]["avg_pressure_error"]
    pressure_error_multi = multi_results["summary"]["overall"]["avg_pressure_error"]
    pressure_improvement = ((pressure_error_single - pressure_error_multi) / pressure_error_single) * 100
    
    # Print summary
    print("\nOVERALL PERFORMANCE COMPARISON:")
    print(f"  Position Error:     Single Agent: {pos_error_single:.2f} km  |  Multi-Agent: {pos_error_multi:.2f} km")
    print(f"                      Improvement: {pos_improvement:.2f}%")
    print(f"  Wind Speed Error:   Single Agent: {wind_error_single:.2f} mph  |  Multi-Agent: {wind_error_multi:.2f} mph")
    print(f"                      Improvement: {wind_improvement:.2f}%")
    print(f"  Pressure Error:     Single Agent: {pressure_error_single:.2f} mb  |  Multi-Agent: {pressure_error_multi:.2f} mb")
    print(f"                      Improvement: {pressure_improvement:.2f}%")
    
    # Early forecast comparison (24h)
    pos_24h_single = single_results["summary"]["overall"]["by_time"][24]["avg_position_error"]
    pos_24h_multi = multi_results["summary"]["overall"]["by_time"][24]["avg_position_error"]
    pos_24h_improvement = ((pos_24h_single - pos_24h_multi) / pos_24h_single) * 100
    
    print("\nEARLY FORECAST (24h) COMPARISON:")
    print(f"  Position Error:     Single Agent: {pos_24h_single:.2f} km  |  Multi-Agent: {pos_24h_multi:.2f} km")
    print(f"                      Improvement: {pos_24h_improvement:.2f}%")
    
    # Check which performs better by basin
    print("\nPERFORMANCE BY BASIN:")
    basins = list(single_results["summary"]["by_basin"].keys())
    for basin in basins:
        basin_name = basin
        if basin == 'NA':
            basin_name = 'North Atlantic'
        elif basin == 'WP':
            basin_name = 'Western Pacific'
        
        single_basin = single_results["summary"]["by_basin"][basin]["avg_position_error"]
        multi_basin = multi_results["summary"]["by_basin"][basin]["avg_position_error"]
        basin_improvement = ((single_basin - multi_basin) / single_basin) * 100
        
        better_system = "Multi-Agent" if multi_basin < single_basin else "Single Agent"
        print(f"  {basin_name}:  {better_system} performs better by {abs(basin_improvement):.2f}%")
    
    print("\nRecommendation: " + 
          ("Multi-Agent System" if pos_improvement > 0 else "Single Agent") + 
          f" is preferred ({abs(pos_improvement):.2f}% {'better' if pos_improvement > 0 else 'worse'} overall position accuracy)")

if __name__ == "__main__":
    compare_agent_performance()