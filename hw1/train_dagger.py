import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import argparse
import pickle
import os
import gym
import load_policy_pytorch

from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import *

mpl.use('Agg')


def load_data_from_file(envname):
    # Load the data
    with open(os.path.join('expert_data', envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)

    # training data
    observations = expert_data['observations']
    # training labels
    actions = expert_data['actions']

    return observations, actions


def get_loaders(observations, actions):
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

    return train_loader, valid_loader, input_shape, output_shape


def run_sim(env, nn, policy_fn, max_steps, num_rollouts, render):

    returns = []
    actions = []
    observations = []

    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:

            # if nn, to tensor
            if nn:
                obs = torch.from_numpy(obs).type(torch.FloatTensor)

            # policy
            action = policy_fn(obs[:])

            # back to numpy if tensor and detach
            if nn:
                action = action.cpu().detach().numpy()
                obs = obs.cpu().detach().numpy()

            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)

            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    return returns, observations


def get_actions(observations, policy_fn):

    actions = []
    for obs in observations:
        actions.append(policy_fn(obs))
    return actions


def main():
    # parse arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert-policy-file', type=str)
    parser.add_argument('--envname', type=str)
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--daggers', type=int, default=5,
                        help='number of dagger iterations')

    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument("--max_timesteps", type=int, default=100)
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of expert roll outs')

    args = parser.parse_args()

    # Set the environment
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    # Set the base expert policy
    base_policy_fn = load_policy_pytorch.load_policy(args.expert_policy_file)

    # set optimizer parameters
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999

    # loss function
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # 1) get initial data
    #
    # 2) init model, train model
    # 3) run model in the gym, get observations
    # 4) run base expert in the gym, using collected observations, store actions
    # 5) augment dataset with observations / actions
    # 6) repeat 2)

    # Main dagger loop
    for diter in range(args.daggers):

        print('DAG iter {}'.format(diter))

        # 1) get initial data
        if diter == 0:
            observations, actions = load_data_from_file(args.envname)

        train_loader, valid_loader, input_shape, output_shape = get_loaders(observations, actions)

        # 2) init model, train model
        # Train neural network model
        #

        # init model
        model = BehaviorClone(input_shape, output_shape)
        # init the opimizer
        optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        # init weights
        model.apply(init_weights)

        # Main epoch loop
        train_loss = []
        valid_loss = []
        for epoch in range(args.epochs):
            t_loss = train_valid(model, train_loader, optimizer, loss_fn, True)
            v_loss = train_valid(model, valid_loader, optimizer, loss_fn, False)

            train_loss.append(t_loss)
            valid_loss.append(v_loss)

        # save the model
        save_checkpoint({'state_dict': model.state_dict()}, 'models/DAG_' + args.envname + '.pt')

        # 3) run model in the gym, get observations
        model.eval()

        dag_returns, dag_obs = run_sim(env, True, model, max_steps, args.num_rollouts, args.render)

        # 4) run base expert in the gym, using collected observations, store actions
        dag_actions = get_actions(dag_obs, base_policy_fn)

        # 5) augment dataset with observations / actions
        observations = np.vstack((observations, dag_obs))
        actions = np.vstack((actions, dag_actions))

    # Plot for last model
    fig = plt.figure()
    plt.plot(train_loss, 'b', label="Training")
    plt.plot(valid_loss, 'r', label="Validation")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('output/' + args.envname + '_DAG_train.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
