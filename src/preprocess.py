import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

class DiffusionDataset(Dataset):
    def __init__(self, dataset_name, split='train', transform=None, max_samples=None):
        self.dataset_name = dataset_name
        self.transform = transform
        
        if dataset_name == 'cifar10':
            self.dataset = load_dataset("uoft-cs/cifar10", split=split)
        elif dataset_name == 'imagenet64':
            self.dataset = load_dataset("gitpull/Imagenet_64x64", split=split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img'] if 'img' in item else item['image']
        label = item['label']
        
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(dataset_name, image_size=64):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    return transform

def create_dataloader(config, split='train'):
    dataset_name = config['dataset']
    batch_size = config['batch_size']
    max_samples = config.get('max_images') if split == 'train' else None
    
    actual_split = split
    if dataset_name == 'imagenet64' and split == 'test':
        actual_split = 'train'  # Use train split for validation when test doesn't exist
        max_samples = 10000  # Limit samples for validation
    
    transform = get_transforms(dataset_name)
    dataset = DiffusionDataset(dataset_name, split=actual_split, transform=transform, max_samples=max_samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader
