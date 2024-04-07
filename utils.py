import torch


def get_device():
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# From N_bins evenly spaced bins between start and end, generate N_samples samples
def sample_bins_uniform(batch_size, N, start, end):
  bins = torch.linspace(start, end, N + 1)
  t = torch.distributions.Uniform(bins[:-1], bins[1:]).sample((batch_size, 1)).reshape(batch_size, N)
  return t

# From N_bins evenly spaced bins between start and end, generate N_samples samples where the probability of choosing a sample from each bin
# is weighted by weights
def sample_piececwise_pdf(weights, N_samples, start, end):
  # Generate a piecewise pdf from weights
  # weights: (batch_size, N_bins)
  batch_size = weights.shape[0]
  N_bins = weights.shape[1]

  # Normalize the weights
  weights = weights + 1e-7
  weights = weights / weights.sum(dim = -1, keepdims=True)
  cdf = torch.cumsum(weights, dim = -1)
  cdf = torch.cat([torch.zeros((weights.shape[0], 1), device=weights.device), cdf], dim = -1)
  
  delta_t = (end - start) / N_bins
  t = torch.linspace(start, end, N_bins + 1, device = weights.device)
  unifs = torch.rand(batch_size, N_samples).to(weights.device)
  
  t1_idx = torch.searchsorted(cdf, unifs, right=True)

  # Index of the bin indicates t_bin to sample from

  cdf_end = cdf.gather(1, t1_idx)
  cdf_start = cdf.gather(1, t1_idx - 1)

  return t[t1_idx - 1] + delta_t * (unifs - cdf_start) / (cdf_end - cdf_start)

  

  
  