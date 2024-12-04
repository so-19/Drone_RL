# Autonomous Drone Stabilization using Deep Reinforcement Learning

## Project Description

This project implements an autonomous drone stabilization system using Deep Q-Network (DQN) reinforcement learning in a PyBullet simulation environment. The primary objective is to train a drone to correct its orientation and maintain a stable hover, starting from an initially inverted position.

## Key Features

- Reinforcement Learning-based drone stabilization
- Custom PyBullet simulation environment
- Deep Q-Network (DQN) learning algorithm
- Adaptive exploration strategy
- Detailed state and reward modeling

## Technical Highlights

### Environment Specifications
- Simulator: PyBullet Physics Engine
- State Space: 13-dimensional vector (position, orientation, velocities)
- Action Space: 4 motor forces with 5 discrete levels

### Learning Approach
- Algorithm: Deep Q-Network (DQN)
- Neural Network Architecture:
  - Input Layer: 13 neurons
  - Hidden Layers: 256 → 256 → 128 neurons
  - Layer Normalization
  - Discretized action selection

### Reward Function Components
- Orientation correction
- Height maintenance
- Velocity minimization
- Stability bonus

## Project Objectives

1. Autonomously correct drone orientation
2. Achieve stable hovering
3. Demonstrate effective reinforcement learning application in drone control

## Performance Metrics

- Orientation correction accuracy
- Hovering stability
- Learning convergence rate
- Reward optimization
