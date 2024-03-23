import torch

def get_device():
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_bins_uniform(batch_size, N, start, end):
  bins = torch.linspace(start, end, N + 1)
  t = torch.distributions.Uniform(bins[:-1], bins[1:]).sample((batch_size, 1)).reshape(batch_size, N)
  z = torch.zeros((batch_size, 1))
  t = torch.cat((z, t), dim = 1)
  return t

def get_camera_to_world(transform_mat):