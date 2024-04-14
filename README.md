# NERF Minecraft
- NERF Paper: https://arxiv.org/pdf/2003.08934.pdf

## Usage

You can either train a model on a dataset or load a model that was previously trained. After loading a model, you can generate a mc schematic or render a rotating gif.

```bash
python main.py {train, load}
```

To train a model on a dataset

```bash
python main.py train --data=./data --scale=1.0 --epochs=10000 --save=./out
```

To load a model
```bash
python main.py load --model=./out ...
```