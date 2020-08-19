import os
import argparse
import torch
from torchvision import datasets, transforms
from vae import FCVAE
from vae import ConvVAE
from net_manager import NetManager


def main():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Load weigths for the VAE, not training the model')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Input batch size for training, 128 by default')

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs to train, 10 by default')

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed, 1 by default')

    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Interval between two logs of the training status')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    train_dataset = datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transforms.ToTensor())

    test_dataset = datasets.MNIST(
        'data',
        train=False,
        transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)

    # model = FCVAE(channels=1, z_dim=2).to(device)
    model = ConvVAE(channels=1, z_dim=2).to(device)

    net_manager = NetManager(
        model,
        device,
        train_loader=train_loader,
        test_loader=test_loader)

    if args.weights is None:
        net_manager.set_writer("vae")
        net_manager.train(args.epochs, log_interval=args.log_interval)

        if not os.path.exists("weights"):
            os.mkdir("weights")

        net_manager.save_net("weights/vae.pth")
    else:
        net_manager.load_net(args.weights)

    net_manager.plot_results(dark_background=True)


if __name__ == "__main__":
    main()
