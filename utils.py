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

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


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
    reward_aggregator = lambda x: x.sum(dim=-1) if reward_aggregation == "all" else x

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in tqdm(range(1, n_training_episodes + 1)):
        saved_log_probs = []
        rewards = []
        states, _ = env.reset()
        states = process_coingame_state(states)
        for t in range(max_t):
            actions, log_probs = [], []
            for state, policy in zip(states, policies):
                action, log_prob = policy.act(state)
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
            policy_loss.append(-log_prob * reward_aggregator(disc_return))
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))

    return np.array(scores)
