# Hurricane Prediction System

An AI-powered hurricane and severe storm trajectory and intensity prediction system using reinforcement learning and ensemble methods.

## Overview

This project uses deep reinforcement learning to predict hurricane and severe storm behavior, including path trajectory, intensity, and pressure changes. The system includes both a single-agent approach and an advanced multi-agent ensemble approach for improved predictions.

### Features

- Prediction of hurricane trajectories up to 5 days in advance
- Wind speed and pressure intensity forecasting
- Global coverage across all major ocean basins
- Basin-specific pattern recognition
- Sea surface temperature and environmental factor integration
- Uncertainty estimation with confidence levels
- Interactive visualization dashboard

## Agent Architecture

The system implements two different approaches to hurricane prediction:

### Single Agent Approach

The original implementation uses a unified reinforcement learning agent that handles all aspects of hurricane prediction simultaneously. This approach uses:

- Deep Q-Learning with a neural network architecture
- Experience replay for stable learning
- Ensemble methods for uncertainty estimation
- Basin-specific parameter adjustments

### Multi-Agent Ensemble Approach

The enhanced implementation divides the hurricane prediction task into specialized agents:

1. **Trajectory Agent**: Focused solely on predicting storm motion and path
   - Specializes in understanding atmospheric steering currents
   - Learns basin-specific movement patterns (e.g., Atlantic recurvature)
   - Optimized for position accuracy metrics

2. **Intensity Agent**: Dedicated to wind speed and pressure prediction
   - Focuses on sea surface temperature relationships
   - Models rapid intensification/weakening cycles
   - Learns from thermodynamic factors

3. **Basin-Specific Agents**: Separate agents trained for each ocean basin
   - North Atlantic (NA) specialist
   - Western Pacific (WP) specialist
   - Eastern Pacific (EP) specialist
   - Each optimized for regional patterns and characteristics

4. **Ensemble Coordinator**: Combines predictions with dynamic weighting
   - Weights agent outputs based on confidence and storm characteristics
   - Adapts weights based on storm phase (developing, mature, dissipating)
   - Handles uncertainty and generates probabilistic forecasts

## Performance Comparison

The multi-agent ensemble approach demonstrates improved performance over the single-agent system:

- Better position accuracy, especially for 24-hour and 48-hour forecasts
- Improved intensity predictions for rapidly intensifying storms
- Enhanced basin-specific prediction accuracy
- More accurate uncertainty estimates

Run the comparison script to see detailed results:

```bash
python compare_agents.py