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
      if i != num_layers_coord - 1:
        self.coordinate_net.append(nn.ReLU())

    self.coordinate_net = nn.Sequential(*self.coordinate_net)

    self.color_net = nn.Linear(256 + 3 * 2 * color_embed_dim, 3)
    self.density_net = nn.Linear(256, 1)

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
    coefs = torch.pow(2, torch.arange(0, self.dim)) * torch.pi
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
    self.t_near = 0
    self.t_far = 10
    self.N = 128
    self.lr = 2e-4

    color_embed_dim = 10
    density_embed_dim = 4
    self.model = SimpleNERF(color_embed_dim, density_embed_dim).to(device)
    self.optim = torch.optim.Adam(self.model.parameters(), self.lr)
    self.device = device

  def compute_color(self, o, d):
    batch_size = d.shape[0]
    # Samples N points randomly from N evenly spaced bins, returns 0 and those N points totalling to N + 1 points
    t = sample_bins_uniform(batch_size, self.N, self.t_near, self.t_far).to(self.device)
    deltas = t[:, :-1] - t[:, 1:]
    # Eliminate starting 0
    t = t[:, 1:]

    t = t.unsqueeze(dim=2)
    d = d.unsqueeze(dim=1)
    o = o.unsqueeze(dim=1)
    # Yields N + 1 points along the ray
    # (batch_size, N + 1, 3)
    x = o + t * d
    d = d.expand(-1, x.shape[1], -1)

    # Convert to (batch_size * N, 3) to pass it in to model
    x = torch.flatten(x, start_dim=0, end_dim=1)
    d = torch.flatten(d, start_dim=0, end_dim=1)
    colors, densitys = self.model(x, d)

    colors = colors.reshape(batch_size, self.N, 3)
    densitys = densitys.reshape(batch_size, self.N)

    alphas = 1 - torch.exp(-densitys * deltas)
    transmittances = torch.cumprod(1 - alphas, dim=1)
    color = torch.sum((transmittances * (1 - alphas)).unsqueeze(2) * colors, dim=1)

    return color.to(device=self.device)

  def train(
    self,
    num_epochs,
    train_data,
    valid_data,
    batch_size=4096,
    img_folder = './results'
  ):
    # print(train_data)
    for epoch in range(num_epochs):
      (rays_d, rays_o, colors) = train_data.sample_rand_img_rays(batch_size)

      (rays_d, rays_o, colors) = (rays_d.to(self.device), rays_o.to(self.device), colors.to(self.device))
      pred_color = self.compute_color(rays_o, rays_d)
      loss = nn.functional.mse_loss(
        pred_color,
        colors,
        reduction="mean",
      )

      self.optim.zero_grad()
      loss.backward()
      self.optim.step()

      print(f"Average loss for epoch {epoch} was {loss.cpu().detach()}")

      if epoch % 200 == 0:
        transform = torch.tensor([
            [
              -0.9938939213752747,
              -0.10829982906579971,
              0.021122142672538757,
              0.08514608442783356,
            ],
            [
              0.11034037917852402,
              -0.9755136370658875,
              0.19025827944278717,
              0.7669557332992554,
            ],
            [
              0.0,
              0.19142703711986542,
              0.9815067052841187,
              3.956580400466919,
            ],
            [
              0.0,
              0.0,
              0.0,
              1.0,
            ],
          ]
        )
        img_tensor = render_img(self, (800, 800), train_data.focal_length, transform)
        save_image(img_tensor, os.path.join(img_folder, f'img_{epoch}.png'))
