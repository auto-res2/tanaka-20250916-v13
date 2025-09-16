#!/usr/bin/env python3
import argparse
import yaml
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .preprocess import create_dataloader
from .train import LACEPlusTrainer
from .evaluate import evaluate_model

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_experiment(config_path, experiment_type):
    print(f"\n{'='*60}")
    print(f"STARTING {experiment_type.upper()} EXPERIMENT")
    print(f"{'='*60}")
    
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    print("\nLoading datasets...")
    train_loader = create_dataloader(config, split='train')
    val_loader = create_dataloader(config, split='test')
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    print("\nInitializing LACE++ trainer...")
    trainer = LACEPlusTrainer(config)
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for epoch in range(config['epochs']):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            loss = trainer.train_step(batch)
            epoch_losses.append(loss)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    print("\nTraining completed. Starting evaluation...")
    
    results = evaluate_model(trainer.unet, val_loader, config, output_dir)
    
    print(f"\n{experiment_type.upper()} EXPERIMENT COMPLETED SUCCESSFULLY!")
    return results

def main():
    parser = argparse.ArgumentParser(description='LACE++ Diffusion Training')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', 
                      help='Run smoke test with reduced parameters')
    group.add_argument('--full-experiment', action='store_true',
                      help='Run full experiment with complete parameters')
    
    args = parser.parse_args()
    
    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
        experiment_type = 'smoke test'
    else:
        config_path = 'config/full_experiment.yaml'
        experiment_type = 'full experiment'
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found!")
        sys.exit(1)
    
    try:
        results = run_experiment(config_path, experiment_type)
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {results.get('sample_image_path', 'N/A')}")
        
    except Exception as e:
        print(f"\nError during experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
