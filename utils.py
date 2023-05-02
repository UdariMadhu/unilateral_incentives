import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch.distributions import Categorical
from tqdm import tqdm


class policy_multiagent(nn.Module):

    def __init__(self, state_dim, hidden_emb, action_dim, device):
        super(policy_multiagent, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_emb)
        self.fc2 = nn.Linear(hidden_emb, action_dim)
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state, action):
        x = torch.from_numpy(np.concatenate(
            [state, action])).float().unsqueeze(0).to(self.device)
        probs = self.forward(x).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class phi_agent(nn.Module):

    def __init__(self, state_dim, hidden_emb, num_actions, device):
        super(phi_agent, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_emb)
        self.fc2 = nn.Linear(hidden_emb, num_actions)
        self.device = device

    def forward(self, s):
        # x1 = torch.from_numpy(np.concatenate([s, [a]], axis=-1)).float().to(self.device)
        x1 = torch.from_numpy(s).float().to(self.device)
        x2 = F.relu(self.fc1(x1))
        x3 = self.fc2(x2)
        return x3


def process_coingame_state(state):
    return np.stack([s.reshape(-1) for s in state])


def reinforce_multiagent(env,
                         policies,
                         optimizer,
                         n_training_episodes,
                         max_t,
                         gamma,
                         print_every,
                         reward_aggregation="self",
                         actionsLiteral=None):
    """
    Policy-gradient for multi-agent learning

    reward_aggregation: "self" or "all" (each agent maximizes its own reward or global reward)
    """
    assert reward_aggregation in ["self", "all"]
    reward_aggregator = lambda x: x.sum(dim=-1
                                        ) if reward_aggregation == "all" else x

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in tqdm(range(1, n_training_episodes + 1)):
        saved_log_probs = []
        rewards = []
        states, _ = env.reset()
        states = process_coingame_state(states)
        prev_action = [0.] * len(policies)
        for t in range(max_t):
            actions, log_probs = [], []
            for state, policy in zip(states, policies):
                action, log_prob = policy.act(state, prev_action)
                actions.append(action)
                log_probs.append(log_prob)
            log_probs = torch.cat(log_probs, dim=-1)
            saved_log_probs.append(log_probs)
            states, _, reward, done = env.step(
                [actionsLiteral[a] for a in actions])
            states = process_coingame_state(states)
            rewards.append(reward)
            if done:
                break
        rewards = np.array(rewards)

        scores_deque.append(np.sum(rewards, axis=0))
        scores.append(np.sum(rewards, axis=0))
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        # Discounting returns for each agent
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(np.array(returns))
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            # policy_loss.append(-log_prob * reward_aggregator(disc_return))
            # policy_loss.append(-log_prob[0] * disc_return.sum(dim=-1) - log_prob[1] * disc_return[1])
            policy_loss.append(-log_prob[0] *
                               (disc_return[0] - disc_return[1]).square() -
                               log_prob[1] * disc_return[1])
        policy_loss = torch.stack(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))

    return np.array(scores)


def reinforce_multiagent_with_phi(env,
                                  policies,
                                  phi_net,
                                  optimizer,
                                  optimizer_phi,
                                  n_training_episodes,
                                  max_t,
                                  gamma,
                                  lambda_phi,
                                  print_every,
                                  reward_aggregation="self",
                                  actionsLiteral=None):
    """
    Policy-gradient for multi-agent learning with phi_agent. We by default assume phi_net
    is applied to first-agent

    reward_aggregation: "self" or "all" (each agent maximizes its own reward or global reward)

    """
    assert reward_aggregation in ["self", "all"]
    reward_aggregator = lambda x: x.sum(dim=-1
                                        ) if reward_aggregation == "all" else x

    scores_deque = deque(maxlen=100)
    scores = []
    optimizer_phi.zero_grad()

    print("Harcoded num_actions = 4")
    num_actions = 4

    for i_episode in tqdm(range(1, n_training_episodes + 1)):
        saved_log_probs = []
        rewards = []
        states, _ = env.reset()
        states = process_coingame_state(states)
        s0, prev_action = states[0], [0.] * len(policies)
        avg_r1, avg_r2, gamma_r = None, None, 0.9  # moving average of rewards
        for t in range(max_t):
            actions = []
            for state, policy in zip(states, policies):
                action, log_prob = policy.act(
                    state, [a / num_actions for a in prev_action])
                actions.append(action)
            saved_log_probs.append(
                log_prob)  # we'll maintain logprobs for second agents only
            prev_action = actions
            # loss function based on first agent policy and phi net
            phi_out0 = phi_net(s0)
            policy_out0_prob = policies[0](torch.from_numpy(
                np.concatenate([s0,
                                [0.] * len(policies)])).float().unsqueeze(0))
            phi_out1 = phi_net(states[0])
            action_t = actions[0]

            states, _, reward, done = env.step(
                [actionsLiteral[a] for a in actions])
            states = process_coingame_state(states)
            rewards.append(reward)

            # phi_net output (hardcoding two agents)
            r1, r2 = reward[0], reward[1]
            if t == 0:
                avg_r1, avg_r2 = r1, r2
            else:
                avg_r1 = gamma_r * avg_r1 + (1 - gamma_r) * r1
                avg_r2 = gamma_r * avg_r2 + (1 - gamma_r) * r2
            phi_out2 = phi_net(states[0])
            policy_out2_prob = policies[0](torch.from_numpy(
                np.concatenate([s0, [a / num_actions
                                     for a in actions]])).float().unsqueeze(0))

            # TODO: Choice between mean and sum
            loss_phi = ((avg_r1 + avg_r2) - phi_out1[action_t] + lambda_phi *
                        (policy_out2_prob * phi_out2).mean() +
                        (1 - lambda_phi) *
                        (phi_out0 * policy_out0_prob).mean()).square()
            # print(avg_r1, avg_r2, phi_out1[action_t].item(), lambda_phi * (policy_out2_prob * phi_out2).mean().item(), 
            #       (1 - lambda_phi) * (phi_out0 * policy_out0_prob).mean().item(), loss_phi.item())

            optimizer_phi.zero_grad()
            optimizer[0].zero_grad()
            loss_phi.backward()
            optimizer_phi.step()
            optimizer[0].step()

            if done:
                break

        # import pdb; pdb.set_trace()
        rewards = np.array(rewards)

        scores_deque.append(np.sum(rewards, axis=0))
        # scores.append(np.sum(rewards, axis=0))
        scores.append(rewards)
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        # Discounting returns for each agent
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(np.array(returns), requires_grad=False)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        returns = returns.detach()

        # saved_log_probs is only for second agent
        if reward_aggregation == "self":
            policy_loss = (-1 * torch.cat(saved_log_probs) *
                           reward_aggregator(returns)[:, -1])
        else:
            policy_loss = (-1 * torch.cat(saved_log_probs) *
                           reward_aggregator(returns))
        if len(policy_loss.shape) == 1:
            policy_loss_final = policy_loss.sum()
        else:
            policy_loss_final = policy_loss[:, -1].sum()

        # harcoded optimization for only second policy network
        assert isinstance(optimizer, list)
        optimizer[1].zero_grad()
        policy_loss_final.backward()
        optimizer[1].step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))

        # save every 50 episodes
        if i_episode % 50 == 0 or i_episode == n_training_episodes:
            yield i_episode, np.stack(scores)
            scores = [
            ]  # to avoid reptitively saving previous episodes rewards
