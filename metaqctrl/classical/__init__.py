"""
Classical control tasks for validating MAML implementation.

Provides simple, fast environments to verify meta-learning works
before applying to quantum systems.
"""

from .pendulum import PendulumEnvironment, PendulumTask
from .task_distribution import PendulumTaskDistribution

__all__ = [
    'PendulumEnvironment',
    'PendulumTask',
    'PendulumTaskDistribution'
]
