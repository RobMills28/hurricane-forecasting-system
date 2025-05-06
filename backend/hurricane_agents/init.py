"""
Hurricane Agents Package

A Python package for hurricane and severe storm trajectory and intensity prediction
using machine learning agent-based models.
"""

__version__ = '0.2.0'

from .base_agent import BaseAgent, QNetwork, ReplayBuffer
from .trajectory_agent import TrajectoryAgent
from .intensity_agent import IntensityAgent
from .basin_specific_agent import BasinSpecificAgent
from .ensemble import EnsembleCoordinator