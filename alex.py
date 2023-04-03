import argparse
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from torch.distributions import Categorical
from tqdm import tqdm

from evoenv.envs import CoinGame

class FeedForward(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, *args):
        x = torch.cat([torch.flatten(arg.float()) for arg in args]).to(self.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Policy(FeedForward):

    def __init__(self, *args):
        super(Policy, self).__init__(*args)

    def forward(self, *args):
        return F.softmax(super(Policy, self).forward(*args), dim=-1)

    def act(self, *args):
        d = Categorical(probs=self.forward(*args).cpu())
        action = d.sample()
        return action.item(), d.log_prob(action)

def reinforce_multiagent_with_psi(env,
                                  policy_enforcer,
                                  policy_pg,
                                  psi_net,
                                  optimizer_enforcer,
                                  optimizer_pg,
                                  optimizer_psi,
                                  n_training_episodes,
                                  max_t,
                                  gamma,
                                  lambda_,
                                  actionsLiteral=None):

    print("Harcoded num_actions = 4")
    num_actions = 4

    criterion = nn.MSELoss()

    actions_one_hot = F.one_hot(torch.tensor([actionsLiteral.index(action) for action in actionsLiteral]))
    null_action = torch.zeros((actions_one_hot.shape[1],))

    states, _ = env.reset()
    null_state = torch.tensor(np.zeros_like(states[0]))

    for i_episode in tqdm(range(1, n_training_episodes + 1)):
        saved_log_probs = []
        rewards = []

        states, _ = env.reset()
        s0, prev_actions = torch.from_numpy(states[0]), None

        avg_r1, avg_r2, gamma_r = None, None, 0.9  # moving average of rewards
        for t in range(max_t):
            s1 = torch.from_numpy(states[0])

            psi_out0 = psi_net(s0)
            policy_out0_prob = policy_enforcer(null_state, *[null_action, null_action], s0).squeeze()
            psi_out1 = psi_net(s1)

            actions = []

            # First agent
            if t == 0:
                action, _ = policy_enforcer.act(null_state, *[null_action, null_action], s0)
            else:
                action, _ = policy_enforcer.act(s1, *[actions_one_hot[action, :] for action in prev_actions], s2)
            actions.append(action)

            # Second agent
            action, log_prob = policy_pg.act(torch.from_numpy(states[1]))
            actions.append(action)

            saved_log_probs.append(log_prob)  # we'll maintain logprobs for second agents only

            action_t = actions[0]

            states, _, reward, done = env.step([actionsLiteral[a] for a in actions])
            rewards.append(reward)

            s2 = torch.from_numpy(states[0])

            r1, r2 = reward[0], reward[1]
            if t == 0:
                avg_r1, avg_r2 = r1, r2
            else:
                # avg_r1 = gamma_r * avg_r1 + (1 - gamma_r) * r1
                # avg_r2 = gamma_r * avg_r2 + (1 - gamma_r) * r2
                avg_r1 = gamma_r * avg_r1 + r1
                avg_r2 = gamma_r * avg_r2 + r2

            prev_actions = actions

            psi_out2 = psi_net(s2)
            policy_out2_prob = policy_enforcer(s1, *[actions_one_hot[action, :] for action in prev_actions], s2).squeeze()

            output = psi_out1[action_t] - lambda_ * torch.dot(policy_out2_prob, psi_out2) - (1 - lambda_) * torch.dot(policy_out0_prob, psi_out0)
            target = torch.tensor(avg_r1 - avg_r2).float()

            loss_psi = criterion(output, target)

            optimizer_psi.zero_grad()
            optimizer_enforcer.zero_grad()

            loss_psi.backward()
            
            optimizer_psi.step()
            optimizer_enforcer.step()

            if done:
                break

        rewards = np.array(rewards)

        returns = deque(maxlen=max_t)
        for t in range(max_t-1, -1, -1):
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t])
        
        yield i_episode, returns[0][0], returns[0][1]
        # yield i_episode, avg_r1, avg_r2
        # yield i_episode, r1, r2

        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(np.array(returns), requires_grad=False)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        returns = returns.detach()

        policy_loss = (torch.cat([log_prob.reshape(1) for log_prob in saved_log_probs]) * returns[:, -1]).sum()

        optimizer_pg.zero_grad()
        policy_loss.backward()
        optimizer_pg.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="./configs.yaml")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--exp_name", type=str, default="temp")

    args = parser.parse_args()
    configs = OmegaConf.load(args.configs)
    # print(configs)
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)

    env = CoinGame(coin_payoffs=np.array(configs.env.reward_matrix), grid_shape=(5, 5), n_coins=1)

    policy_enforcer = Policy(
                            configs.env.state_dim + configs.env.num_agents*configs.env.num_actions + configs.env.state_dim,
                            configs.optim.hidden_dim,
                            configs.env.num_actions,
                            configs.device
                        ).to(configs.device)

    policy_pg = Policy(
                    configs.env.state_dim,
                    configs.optim.hidden_dim,
                    configs.env.num_actions,
                    configs.device
                ).to(configs.device)
    
    psi_net = FeedForward(
                        configs.env.state_dim,
                        configs.optim.hidden_dim,
                        configs.env.num_actions,
                        configs.device
                    ).to(configs.device)

    optimizer_enforcer = optim.Adam(policy_enforcer.parameters(), lr=configs.optim.lr)
    optimizer_pg = optim.Adam(policy_pg.parameters(), lr=configs.optim.lr)
    optimizer_psi = optim.Adam(psi_net.parameters(), lr=configs.optim.lr)

    hor, r1, r2 = [], [], []

    for episode, reward1, reward2 in reinforce_multiagent_with_psi(
                    env,
                    policy_enforcer,
                    policy_pg,
                    psi_net,
                    optimizer_enforcer,
                    optimizer_pg,
                    optimizer_psi,
                    configs.optim.num_episodes,
                    configs.optim.steps_per_episode,
                    configs.optim.gamma,
                    configs.optim.lambda_phi,
                    actionsLiteral=configs.env.actionsLiteral):
        pass
        
        hor.append(episode)
        r1.append(reward1)
        r2.append(reward2)

    plt.plot(hor, r1)
    plt.plot(hor, r2)
    plt.show()