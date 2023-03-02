# Ref: https://huggingface.co/blog/deep-rl-pg
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from minenv.envs import CoinGame
from utils import policy_multiagent, reinforce_multiagent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="./configs.yaml")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--exp_name", type=str, default="temp")

    args = parser.parse_args()
    configs = OmegaConf.load(args.configs)
    print(configs)

    env = CoinGame(coin_payoffs=np.array(configs.env.reward_matrix))

    policy_nets = [
        policy_multiagent(configs.env.state_dim, 
                          configs.optim.hidden_dim,
                          configs.env.num_actions,
                          configs.device).to(configs.device)
        for _ in range(configs.env.num_agents)
    ]
    # single optimizer for all agents
    policy_optimizer = optim.Adam(
        [p for net in policy_nets for p in net.parameters()], lr=configs.optim.lr)

    rewards = reinforce_multiagent(
        env,
        policy_nets,
        policy_optimizer,
        configs.optim.num_episodes,
        configs.optim.steps_per_episode,
        configs.optim.gamma,
        print_every=100,
        reward_aggregation=configs.optim.reward_aggregation,
        actionsLiteral=configs.env.actionsLiteral)
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"configs": configs, "rewards": rewards, "ckpt": [net.state_dict() for net in policy_nets]}, 
               f"{args.output_dir}/{args.exp_name}.pt")
    

if __name__ == "__main__":
    main()