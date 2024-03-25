from simple_nerf import SimpleNERFModel
from data import load_data
from utils import *
if __name__ == '__main__':
  data = load_data('./data/hotdog')
  nerf = SimpleNERFModel(device=get_device())
  nerf.train(100, data['train'], data['val'])
  