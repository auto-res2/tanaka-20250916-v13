import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List
from PIL import Image
import torchvision.transforms as transforms

def ddim_sample(model, shape, timesteps=50, device='cuda'):
    model.eval()
    with torch.no_grad():
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(0, timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            eps_pred = model(x, t)
            
            alpha_t = torch.tensor(1 - i / timesteps, device=device)
            alpha_t_prev = torch.tensor(1 - (i-1) / timesteps if i > 0 else 1, device=device)
            
            x = torch.sqrt(alpha_t_prev) * (x - torch.sqrt(1-alpha_t) * eps_pred) / torch.sqrt(alpha_t) + torch.sqrt(1-alpha_t_prev) * eps_pred
            
    return x

def compute_fid_proxy(real_features, fake_features):
    mu_real, sigma_real = real_features.mean(0), np.cov(real_features.T)
    mu_fake, sigma_fake = fake_features.mean(0), np.cov(fake_features.T)
    
    diff = mu_real - mu_fake
    covmean = np.sqrt(sigma_real @ sigma_fake)
    
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def evaluate_model(model, dataloader, config, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    num_samples = 128
    sample_shape = (num_samples, 3, 32, 32) if config['dataset'] == 'cifar10' else (num_samples, 3, 64, 64)
    
    print("Generating samples for evaluation...")
    generated_samples = ddim_sample(model, sample_shape, device=device)
    
    generated_samples = (generated_samples + 1) / 2
    generated_samples = torch.clamp(generated_samples, 0, 1)
    
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    for i in range(64):
        row, col = i // 8, i % 8
        img = generated_samples[i].cpu().permute(1, 2, 0).numpy()
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    sample_path = f"{output_dir}/images/generated_samples.png"
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    fake_features = generated_samples.view(num_samples, -1).cpu().numpy()
    
    real_images = []
    for batch_idx, (images, _) in enumerate(dataloader):
        real_images.append(images)
        if len(real_images) * images.shape[0] >= num_samples:
            break
    
    real_images = torch.cat(real_images)[:num_samples]
    real_features = real_images.view(num_samples, -1).numpy()
    
    fid_score = compute_fid_proxy(real_features, fake_features)
    
    diversity_score = np.std(fake_features.mean(axis=1))
    coverage_score = np.mean(np.std(fake_features, axis=0))
    
    results = {
        "experiment_name": config['experiment_name'],
        "dataset": config['dataset'],
        "fid_score": float(fid_score),
        "diversity_score": float(diversity_score),
        "coverage_score": float(coverage_score),
        "num_samples": num_samples,
        "sample_image_path": sample_path,
        "model_parameters": {
            "epochs": config['epochs'],
            "batch_size": config['batch_size'],
            "lambda_d": config['lace']['lambda_d'],
            "gamma_energy": config['lace']['gamma_energy']
        }
    }
    
    results_path = f"{output_dir}/experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"Experiment: {results['experiment_name']}")
    print(f"Dataset: {results['dataset']}")
    print(f"FID Score: {results['fid_score']:.3f}")
    print(f"Diversity Score: {results['diversity_score']:.3f}")
    print(f"Coverage Score: {results['coverage_score']:.3f}")
    print(f"Generated Samples: {results['num_samples']}")
    print(f"Sample Image Path: {results['sample_image_path']}")
    print(f"Results JSON Path: {results_path}")
    print("="*50)
    
    print("\nJSON RESULTS:")
    print(json.dumps(results, indent=2))
    
    return results
