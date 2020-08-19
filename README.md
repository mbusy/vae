# VAE
Variational Autoencoder implementation for Pytorch. This implementation uses the MNIST dataset, but is compatible with other datasets, such as CIFAR-10 (different dimensions and number of channels).

## Models
The current implentation proposes 2 basic VAE models, a fully connected one (`FCVAE` class) and a convolutional one (`ConvVAE` class). The `NetManager` class can be used to train these models, and to produce logs (tensorboard) / plot results.

## Usage
Using `main.py`:
```shell
# default launch
python3 main.py

# training
python3 main.py --batch-size 128 --epochs 10 --seed 1 --log-interval

# loading existing weights
python3 main.py --weights weights/vae.pth
``` 