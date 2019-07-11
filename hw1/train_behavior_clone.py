import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import argparse
import pickle
import os

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def save_checkpoint(state, filename):
    torch.save(state, filename)


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_weights(sub_mod)


# Model definition
class BehaviorClone(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(BehaviorClone, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.fc1 = nn.Linear(input_shape, input_shape // 2)
        self.fc2 = nn.Linear(input_shape // 2, output_shape)
        self.do = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = self.fc2(x)
        return x


def train_valid(model, loader, optimizer, loss_fn, train):

    # train / valid loop
    model.train() if train else model.eval()
    batch_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        batch_size = x.size(0)
        loss = 0
        if train:
            optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        batch_loss += loss.item() / batch_size

        if train:
            loss.backward()
            optimizer.step()

    batch_loss /= (batch_idx + 1)
    return batch_loss


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
