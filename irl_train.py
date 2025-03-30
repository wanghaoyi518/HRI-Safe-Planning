# main.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from dynamics import CarDynamics
from data_generation import DataGenerator
from irl import MaxEntIRL, StateParameterizedIRL
from features import (
    speed_feature, lane_keeping_feature, 
    collision_avoidance_feature, control_smoothness_feature
)

def main():
    """Main training pipeline."""
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create feature functions
    features = [
        speed_feature,
        lane_keeping_feature,
        collision_avoidance_feature,
        control_smoothness_feature
    ]
    
    # Generate or load dataset
    generator = DataGenerator(dynamics)
    
    # Option 1: Generate new dataset
    print("Generating demonstration dataset...")
    dataset = generator.generate_dataset(num_scenarios=6, episodes_per_scenario=3)
    generator.save_dataset(dataset, "irl_dataset.pkl")
    
    # Option 2: Load existing dataset
    # dataset = generator.load_dataset("irl_dataset.pkl")
    print(f"Dataset contains {len(dataset)} trajectories")
    
    # Train state-parameterized IRL model
    print("Training IRL models...")
    irl = StateParameterizedIRL(features)
    models = irl.train(dataset, num_bins=2)
    
    # Display learned models
    print("\nLearned models:")
    for bin_key, model in models.items():
        print(f"Bin {bin_key}:")
        print(f"  Attentiveness range: {model['att_range']}")
        print(f"  Driving style range: {model['style_range']}")
        print(f"  Weights: {model['weights']}")
    
    # Visualize weight landscape
    print("\nPlotting weight landscape...")
    fig = irl.plot_weight_landscape()
    fig.savefig("weight_landscape.png")
    
    # Test creating a reward function
    test_state = [0.75, 0.25]  # High attentiveness, somewhat conservative
    print(f"\nCreating reward function for internal state {test_state}...")
    reward_fn = irl.create_reward_function(test_state)
    
    print("Done! Trained models can now be used for belief updates and planning.")

if __name__ == "__main__":
    main()