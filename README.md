# Synthetic Time Series Generation using Generative Adversarial Network in Smart Grid
Code that replicate work [Generative Adversarial Network for Synthetic Time Series Data Generation in Smart Grids](https://drive.google.com/file/d/18M0jW2aZDv3dzUfUDoFqn4QKleNX0VAZ/view).

Note that in this repo, we change our model to [ACGAN](https://arxiv.org/abs/1610.09585) instead of the original Conditional GAN.

Focus on periodic time series with daily, weekly and yearly patterns including load and PV generation.

## Package Requirements
```
tqdm==4.30.0
numpy==1.16.2
Keras==2.2.4
tensorboardX==1.6
tensorflow==1.15
tensorboard==1.13.0
```

## Usage
Example: 

- To train a model using Pecan Street Dataset for user with id 171, run

`python main.py --train --num_epoch 100 --id 171`

- To generate synthetic data, run

`python main.py --id 171`

## License
MIT
