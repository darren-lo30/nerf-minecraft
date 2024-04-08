from torch import nn
import torch
from utils import sample_bins_uniform, sample_piececwise_pdf
from render import render_img
from torchvision.utils import save_image
import os
from itertools import chain



class NERFNet(nn.Module):
  def __init__(self, color_embed_dim, density_embed_dim, num_layers_coord=8, skip_connections=[4]):
    super().__init__()
    # If no skip connections, then we can get stuck at an all black image
    self.skip_connections = skip_connections

    self.coordinate_net = nn.ModuleList()
    for i in range(num_layers_coord):
      num_in = 256 if i != 0 else density_embed_dim * 3 * 2
      if i - 1 in skip_connections:
        num_in += density_embed_dim * 3 * 2

      self.coordinate_net.append(nn.Linear(num_in, 256))

    self.color_net = nn.Sequential(nn.Linear(256 + 3 * 2 * color_embed_dim, 256), nn.Linear(256, 3), nn.Sigmoid())
    self.density_net = nn.Sequential(nn.Linear(256, 1), nn.ReLU())

    self.color_embeds = PositionalEmbedding(color_embed_dim)
    self.density_embed = PositionalEmbedding(density_embed_dim)

  def forward(self, x, d):
    x = self.density_embed(x)
    skip = x
    for i, m in enumerate(self.coordinate_net):
      x = m(x)
      x = nn.functional.relu(x)

      if i in self.skip_connections:
        x = torch.cat([x, skip], dim=-1)

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
    x = coefs.to(x.device) * x.unsqueeze(dim=2)

    # batch_size x 3 x 2L
    embeddings = torch.zeros((x.shape[0], 3, self.dim * 2), device=x.device)
    embeddings[:, :, ::2] = torch.sin(x)
    embeddings[:, :, 1::2] = torch.cos(x)

    embeddings = torch.flatten(embeddings, start_dim=1)
    return embeddings


class NERF:
  def __init__(self, device, use_fine=True):
    # Hyperparameters
    self.t_near = 2
    self.t_far = 6
    self.N_fine = 128
    self.N_coarse = 64
    self.lr = 5e-4

    self.use_fine = use_fine

    color_embed_dim = 6
    density_embed_dim = 6
    self.coarse_model = NERFNet(color_embed_dim, density_embed_dim).to(device)
    if use_fine:
      self.fine_model = NERFNet(color_embed_dim, density_embed_dim).to(device)
      params = chain(self.coarse_model.parameters(), self.fine_model.parameters())
    else:
      params = self.coarse_model.parameters()


    self.optim = torch.optim.Adam(params, self.lr)
    self.device = device

  def get_model(self):
    if self.use_fine:
      return self.fine_model
    
    return self.coarse_model

  def compute_color(self, model, t, o, d):
    batch_size = d.shape[0]
    deltas = t[:, 1:] - t[:, :-1]  # (batch_size, N - 1)
    deltas = torch.cat([deltas, torch.tensor([1e10], device=self.device).expand(t.shape[0], -1)], dim=1)  # (batch_size, N)

    N = t.shape[1]
    t = t.unsqueeze(dim=2)  # (batch_size, N, 1)
    d = d.unsqueeze(dim=1)  # (batch_size, 1, 3)
    o = o.unsqueeze(dim=1)  # (batch_size, 1, 3)

    # (batch_size, N, 3)
    x = o + t * d
    d = d.expand(-1, x.shape[1], -1)  # (batchsize, N, 3)

    # Convert to (batch_size * N, 3) to pass it in to model
    x = torch.flatten(x, start_dim=0, end_dim=1)
    d = torch.flatten(d, start_dim=0, end_dim=1)
    colors, densitys = model(x, d)

    colors = colors.reshape(batch_size, N, 3)
    densitys = densitys.reshape(batch_size, N)
    alphas = 1 - torch.exp(-densitys * deltas)
    transmittances = torch.cumprod(1 - alphas + 1e-10, dim=1)
    weights = transmittances * alphas
    color = torch.sum(weights.unsqueeze(2) * colors, dim=1)

    return color.to(device=self.device), weights

  def train(self, num_epochs, train_data, batch_size=4096, img_folder="./results"):
    img_dims = train_data.imgs[0].shape[1:3]

    final_lr = 5e-5
    decay = (final_lr / self.lr) ** (1.0 / num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, decay)

    for epoch in range(num_epochs):
      if epoch % (num_epochs // 3) == 0 or epoch == 50:
        transform = torch.tensor(
          [
            [6.8935126e-01, 5.3373039e-01, -4.8982298e-01, -1.9745398e00],
            [-7.2442728e-01, 5.0788772e-01, -4.6610624e-01, -1.8789345e00],
            [1.4901163e-08, 6.7615211e-01, 7.3676193e-01, 2.9699826e00],
            [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
          ]
        )
        img_tensor = render_img(self, img_dims, train_data.focal_length, transform)
        save_image(img_tensor, os.path.join(img_folder, f"img_{epoch}.png"))

      (rays_d, rays_o, colors) = train_data.sample_rand_img_rays(batch_size)

      (rays_d, rays_o, colors) = (rays_d.to(self.device), rays_o.to(self.device), colors.to(self.device))

      # Samples N points randomly from N evenly spaced bins, returns 0 and those N points totalling to N + 1 points
      t_coarse = sample_bins_uniform(batch_size, self.N_coarse, self.t_near, self.t_far).to(self.device)  # (batch_size, N)

      coarse_pred_color, weights = self.compute_color(self.coarse_model, t_coarse, rays_o, rays_d)
      coarse_loss = nn.functional.mse_loss(
        colors,
        coarse_pred_color,
        reduction="mean",
      )

      if self.use_fine:
        t_fine = sample_piececwise_pdf(weights.detach(), self.N_fine, self.t_near, self.t_far).to(self.device)
        fine_pred_color, _ = self.compute_color(self.fine_model, t_fine, rays_o, rays_d)

        fine_loss = nn.functional.mse_loss(
          colors,
          fine_pred_color,
          reduction="mean",
        )

        loss = coarse_loss + fine_loss
      else:
        loss = coarse_loss

      self.optim.zero_grad()
      loss.backward()
      self.optim.step()
      lr_scheduler.step()

      print(f"Average loss for epoch {epoch} was {loss.cpu().detach()}")

  def save(self, path):
    data = {
      'coarse': self.coarse_model.state_dict(),
      'use_fine': self.use_fine
    }
    if self.use_fine:
      data['fine'] = self.fine_model.state_dict(),

    torch.save(data, path)

  def load(self, path):
    data = torch.load(path)
    self.coarse_model.load_state_dict(data['coarse'])
    self.coarse_model.eval()
    
    self.use_fine = data['use_fine']
    if self.use_fine:
      self.fine_model.load_state_dict(data['fine'])
      self.fine_model.eval()
