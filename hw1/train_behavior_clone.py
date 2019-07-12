import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import argparse
import pickle
import os

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import *


def main():
    # parse arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str)
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of training epochs')

    args = parser.parse_args()

    # Load the data
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)

    # training data
    observations = expert_data['observations']
    # training labels
    actions = expert_data['actions']

    # split
    X_train, X_valid, y_train, y_valid = train_test_split(observations, actions, test_size=0.33)

    X_train = torch.from_numpy(X_train).type(torch.torch.FloatTensor)
    X_valid = torch.from_numpy(X_valid).type(torch.torch.FloatTensor)

    y_train = torch.from_numpy(y_train).squeeze(1).type(torch.torch.FloatTensor)
    y_valid = torch.from_numpy(y_valid).squeeze(1).type(torch.torch.FloatTensor)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_valid, y_valid)

    batch_size = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, pin_memory=True)

    valid_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, pin_memory=True)

    input_shape = X_train.size(1)
    output_shape = y_valid.size(-1)

    # init model
    model = BehaviorClone(input_shape, output_shape)

    # init weights
    model.apply(init_weights)

    # set optimizer
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Main epoch loop
    train_loss = []
    valid_loss = []
    for epoch in range(args.epochs):
        t_loss = train_valid(model, train_loader, optimizer, loss_fn, True)
        v_loss = train_valid(model, valid_loader, optimizer, loss_fn, False)

        train_loss.append(t_loss)
        valid_loss.append(v_loss)

    # save the model
    save_checkpoint({'state_dict': model.state_dict()}, 'models/BC_' + args.envname + '.pt')

    # plot here
    fig = plt.figure()
    plt.plot(train_loss, 'b', label="Training")
    plt.plot(valid_loss, 'r', label="Validation")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('output/' + args.envname + '_behavior_clone_train.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
