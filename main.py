from simple_nerf import SimpleNERFModel
from data import load_data
from utils import get_device

if __name__ == "__main__":
  data = load_data("./data/lego", scale_ratio=0.1)
  nerf = SimpleNERFModel(device=get_device())
  # 100 train images
  # ~ 1000 epochs (over 100 images)
  num_epochs = 10000
  nerf.train(num_epochs, data["train"], data["val"])
  # x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
  # print(x.swapaxes(0, 1)) 