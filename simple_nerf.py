from torch import nn
import torch
from utils import *

# Basic NERF with only one sampling method
class SimpleNERF(nn.Module):
  def __init__(self, color_embed_dim, density_embed_dim, num_layers_coord = 8):
    super().__init__()
    self.coordinate_net = nn.ModuleList()
    for i in range(num_layers_coord):
      num_in = 256 if i != 0 else color_embed_dim * 3
      self.coordinate_net.append(nn.Linear(num_in, 256))
      if i != num_layers_coord - 1:
        self.coordinate_net.append(nn.ReLU())

    self.coordinate_net = nn.Sequential(self.coordinate_net)

    self.color_net = nn.Linear(256 + 3 * density_embed_dim, 3)
    self.density_net = nn.Linear(256, 1)

    self.color_embeds = PositionalEmbedding(color_embed_dim)
    self.density_embed = PositionalEmbedding(color_embed_dim)

  def forward(self, x, d):
    x = self.pos_embeds(x)
    x = self.coordinate_net(x)

    density = self.density_net(x)

    # Concat position with viewing direction
    x = torch.cat(x, d)
    color = self.color_net(x)

    return color, density


class PositionalEmbedding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    x = torch.pow(2, torch.arange(0, self.dim)) * torch.pi * x
    # batch_size x 6 x L
    embeddings = torch.zeros((x.shape[0], 6, self.dim))
    embeddings[:,:,::2] = torch.sin(x)
    embeddings[:,:,1::2] = torch.cos(x)

    return embeddings


def compute_color(model, o, d, t_near, t_far, N):
  batch_size = o.shape[0]
  t = sample_bins_uniform(batch_size, t_near, t_far, N)
  deltas = t[:-1] - t[1:]

  x = o + t * d
  colors, densitys = model(x, d)

  alphas = 1 - torch.exp(-densitys * deltas)
  transmittances = torch.cumprod(1 - alphas)

  color = torch.sum(transmittances * (1 - alphas) * colors)

  return color


def train(model, num_epochs):
  for epoch in range(num_epochs)
