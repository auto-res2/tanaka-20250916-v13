import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import math

class SimSiamProjector(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        return self.projector(x).squeeze(-1).squeeze(-1)

class CriticSampler(nn.Module):
    def __init__(self, K=100, S=10, input_channels=7):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1), 
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, 1, 1), 
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.w_head = nn.Linear(16, 2)
        self.pi_head = nn.Linear(16, K * S)
        self.K, self.S = K, S
        
    def forward(self, x_t, eps_hat, t_emb):
        h = self.backbone(torch.cat([x_t, eps_hat, t_emb.expand_as(x_t[:,:1])], 1))
        h = h.squeeze(-1).squeeze(-1)
        
        w_out = self.w_head(h).sigmoid() * 9 + 1e-1
        w_q, w_d = w_out[:, 0], w_out[:, 1]
        pi = self.pi_head(h)
        return w_q, w_d, pi.view(-1, self.K, self.S)

class SimpleUNet(nn.Module):
    def __init__(self, channels=3, model_channels=64, num_res_blocks=2):
        super().__init__()
        self.channels = channels
        self.model_channels = model_channels
        
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        
        self.input_conv = nn.Conv2d(channels, model_channels, 3, 1, 1)
        self.down1 = nn.Conv2d(model_channels, model_channels * 2, 3, 2, 1)
        self.down2 = nn.Conv2d(model_channels * 2, model_channels * 4, 3, 2, 1)
        
        self.middle = nn.Conv2d(model_channels * 4, model_channels * 4, 3, 1, 1)
        
        self.up2 = nn.ConvTranspose2d(model_channels * 4, model_channels * 2, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, 2, 1)
        self.output_conv = nn.Conv2d(model_channels, channels, 3, 1, 1)
        
    def forward(self, x, t, y=None):
        t_emb = self.timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        h = self.input_conv(x)
        h1 = self.down1(h)
        h2 = self.down2(h1)
        h = self.middle(h2)
        h = self.up2(h) + h1
        h = self.up1(h) + self.input_conv(x)
        return self.output_conv(h)
    
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod = torch.sqrt(1 - t.float() / 1000)[:, None, None, None]
    sqrt_one_minus_alphas_cumprod = torch.sqrt(t.float() / 1000)[:, None, None, None]
    
    return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

class LACEPlusTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.unet = SimpleUNet(channels=3, model_channels=config['model']['unet_channels']).to(self.device)
        self.projector = SimSiamProjector().to(self.device)
        self.critic = CriticSampler(
            K=config['lace']['K_bins'], 
            S=config['lace']['S_strata']
        ).to(self.device)
        
        self.unet_optimizer = torch.optim.AdamW(self.unet.parameters(), lr=config['optimizer']['lr'])
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config['lace']['critic_lr'])
        
        self.step = 0
        self.meta_buffer = []
        
    def train_step(self, batch):
        x0, labels = batch
        x0 = x0.to(self.device)
        B = x0.shape[0]
        
        t = torch.randint(0, self.config['model']['timesteps'], (B,), device=self.device)
        
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, noise)
        
        eps_hat = self.unet(x_t, t)
        
        t_emb = self.unet.timestep_embedding(t, self.unet.model_channels)
        w_q, w_d, log_pi = self.critic(x_t, eps_hat.detach(), t_emb[:, :1, None, None])
        
        lambda_d = self.config['lace']['lambda_d']
        loss = (w_q + lambda_d * w_d) * F.mse_loss(noise, eps_hat, reduction='none').mean(dim=[1,2,3])
        loss = loss.mean()
        
        self.unet_optimizer.zero_grad()
        loss.backward()
        self.unet_optimizer.step()
        
        self.meta_buffer.append({
            'loss': loss.item(),
            'w_q': w_q.mean().item(),
            'w_d': w_d.mean().item()
        })
        
        if self.step % self.config['lace']['meta_interval'] == 0 and len(self.meta_buffer) > 0:
            self.meta_update()
            
        self.step += 1
        return loss.item()
    
    def meta_update(self):
        if len(self.meta_buffer) < 10:
            return
            
        recent_losses = [item['loss'] for item in self.meta_buffer[-10:]]
        reward = -np.mean(recent_losses)
        
        self.critic_optimizer.zero_grad()
        self.critic_optimizer.step()
        
        self.meta_buffer = []
