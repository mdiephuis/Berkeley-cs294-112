#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)

*** behavior cloning ***

"""

import os
import pickle
import numpy as np
import torch
import gym
import load_policy_pytorch
import argparse
from utils import BehaviorClone

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')


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

    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert-policy-file', type=str)
    parser.add_argument('--envname', type=str)
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument("--max_timesteps", type=int, default=100)
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of expert roll outs')

    args = parser.parse_args()

    # Base policy for comparison
    print('loading and building expert policy')
    base_policy_fn = load_policy_pytorch.load_policy(args.expert_policy_file)
    print('loaded and built')

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    # Get environment datasizes
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)

    # load behavior clone model
    input_size = expert_data['observations'].shape[-1]
    output_size = expert_data['actions'].shape[-1]
    bclone_model = BehaviorClone(input_size, output_size)

    # Read pre-trained weights from disk
    checkpoint = torch.load('models/DAG_' + args.envname + '.pt', map_location='cpu')
    bclone_model.load_state_dict(checkpoint['state_dict'])

    # Set to eval
    bclone_model.eval()

    # Run simulation
    base_returns = run_sim(env, False, base_policy_fn, max_steps, args.num_rollouts, args.render)
    bclone_returns = run_sim(env, True, bclone_model, max_steps, args.num_rollouts, args.render)

    # plot here
    fig = plt.figure()
    plt.plot(base_returns, 'b', label="Base")
    plt.plot(bclone_returns, 'r', label="DAG Behavior clone")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('output/' + args.envname + '_dag_rewards.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
