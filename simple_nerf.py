from torch import nn
import torch
from utils import sample_bins_uniform
from render import render_img
from torchvision.utils import save_image
import os

# Basic NERF with only one sampling method
class SimpleNERF(nn.Module):
  def __init__(
    self,
    color_embed_dim,
    density_embed_dim,
    num_layers_coord=8,
  ):
    super().__init__()
    self.coordinate_net = nn.ModuleList()
    for i in range(num_layers_coord):
      num_in = 256 if i != 0 else density_embed_dim * 3 * 2
      self.coordinate_net.append(nn.Linear(num_in, 256))
      self.coordinate_net.append(nn.ReLU())

    self.coordinate_net = nn.Sequential(*self.coordinate_net)

    self.color_net = nn.Sequential(nn.Linear(256 + 3 * 2 * color_embed_dim, 256), nn.Linear(256, 3), nn.Sigmoid())
    self.density_net = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1), nn.ReLU())

    self.color_embeds = PositionalEmbedding(color_embed_dim)
    self.density_embed = PositionalEmbedding(density_embed_dim)

  def forward(self, x, d):
    x = self.density_embed(x)
    x = self.coordinate_net(x)
    density = self.density_net(x)

    # Concat position with viewing direction
    d = self.color_embeds(d)
    x = torch.cat((x, d), dim=1)

    color = self.color_net(x)

    return (color, density)


class PositionalEmbedding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    coefs = torch.pow(2, torch.arange(0, self.dim)) 
    x = coefs.to(x.get_device()) * x.unsqueeze(dim=2)

    # batch_size x 3 x 2L
    embeddings = torch.zeros((x.shape[0], 3, self.dim * 2), device=x.get_device())
    embeddings[:, :, ::2] = torch.sin(x)
    embeddings[:, :, 1::2] = torch.cos(x)

    embeddings = torch.flatten(embeddings, start_dim=1)
    return embeddings


class SimpleNERFModel:
  def __init__(self, device):
    # Hyperparameters
    self.t_near = 2
    self.t_far = 10
    self.N = 64
    self.lr = 5e-4


    color_embed_dim = 6
    density_embed_dim = 6
    self.model = SimpleNERF(color_embed_dim, density_embed_dim).to(device)
    self.optim = torch.optim.Adam(self.model.parameters(), self.lr)
    # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.99)
    self.device = device

  def compute_color(self, o, d):
    batch_size = d.shape[0]
    # Samples N points randomly from N evenly spaced bins, returns 0 and those N points totalling to N + 1 points
    t = sample_bins_uniform(batch_size, self.N, self.t_near, self.t_far).to(self.device) # (batch_size, N)
    deltas = t[:, 1:] - t[:, :-1] # (batch_size, N - 1)
    deltas = torch.cat([deltas, torch.tensor([1e10],  device=self.device).expand(t.shape[0], -1)], dim = 1)  # (batch_size, N)

    t = t.unsqueeze(dim=2) # (batch_size, N, 1)
    d = d.unsqueeze(dim=1) # (batch_size, 1, 3)
    o = o.unsqueeze(dim=1) # (batch_size, 1, 3)

    # (batch_size, N, 3)
    x = o + t * d
    d = d.expand(-1, x.shape[1], -1) #(batchsize, N, 3)

    # Convert to (batch_size * N, 3) to pass it in to model
    x = torch.flatten(x, start_dim=0, end_dim=1)
    d = torch.flatten(d, start_dim=0, end_dim=1)
    colors, densitys = self.model(x, d)

    colors = colors.reshape(batch_size, self.N, 3)
    densitys = densitys.reshape(batch_size, self.N)
    alphas = 1 - torch.exp(-densitys * deltas)
    transmittances = torch.cumprod(1 - alphas + 1e-10, dim=1)
    color = torch.sum((transmittances * alphas).unsqueeze(2) * colors, dim=1)

    return color.to(device=self.device)

  def train(
    self,
    num_epochs,
    train_data,
    valid_data,
    batch_size=6400,
    img_folder = './results'
  ):
    for epoch in range(num_epochs):
      (rays_d, rays_o, colors) = train_data.sample_rand_img_rays(batch_size)

      (rays_d, rays_o, colors) = (rays_d.to(self.device), rays_o.to(self.device), colors.to(self.device))
      pred_color = self.compute_color(rays_o, rays_d)

      loss = nn.functional.mse_loss(
        colors,
        pred_color,
        reduction="mean",
      )
      # if epoch % 20 == 0:
      #   save_image(pred_color.transpose(0, 1).reshape(3, 80, 80), os.path.join('./results', f'img_test_{epoch}.png'))

      self.optim.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
      self.optim.step()
      # self.lr_scheduler.step()
      # print(self.lr_scheduler.get_lr())

      print(f"Average loss for epoch {epoch} was {loss.cpu().detach()}")

      if epoch % 25 == 0:
        transform = torch.tensor([[ 6.8935126e-01, 5.3373039e-01, -4.8982298e-01, -1.9745398e+00],
                                  [-7.2442728e-01, 5.0788772e-0, -4.6610624e-01, -1.8789345e+00],
                                  [ 1.4901163e-08, 6.7615211e-01, 7.3676193e-01, 2.9699826e+00],
                                  [ 0.0000000e+00, 0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]
        )
        img_tensor = render_img(self, (80, 80), train_data.focal_length, transform)
        save_image(img_tensor, os.path.join(img_folder, f'img_{epoch}.png'))
