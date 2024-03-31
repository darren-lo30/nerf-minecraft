import torch

def get_device():
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_bins_uniform(batch_size, N, start, end):
  bins = torch.linspace(start, end, N + 1)
  t = torch.distributions.Uniform(bins[:-1], bins[1:]).sample((batch_size, 1)).reshape(batch_size, N)
  return t
