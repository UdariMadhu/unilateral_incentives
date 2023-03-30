# Ref: https://huggingface.co/blog/deep-rl-pg
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from minenv.envs import CoinGame
from utils import policy_multiagent, reinforce_multiagent_with_phi, phi_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="./configs.yaml")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--exp_name", type=str, default="temp")

    args = parser.parse_args()
    configs = OmegaConf.load(args.configs)
    print(configs)
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)

    env = CoinGame(coin_payoffs=np.array(configs.env.reward_matrix), grid_shape=(5, 5))

    policy_nets = [
        policy_multiagent(configs.env.state_dim + configs.env.num_agents, configs.optim.hidden_dim,
                          configs.env.num_actions,
                          configs.device).to(configs.device)
        for _ in range(configs.env.num_agents)
    ]
    phi_net = phi_agent(configs.env.state_dim, configs.optim.hidden_dim, 
                        configs.env.num_actions,
                        configs.device).to(configs.device)

    # single optimizer for all agents
    # policy_optimizers = optim.Adam(
    #     [p for net in policy_nets for p in net.parameters()], lr=configs.optim.lr)
    policy_optimizers = [
        optim.Adam(net.parameters(), lr=configs.optim.lr)
        for net in policy_nets
    ]
    phi_optimizer = optim.Adam(phi_net.parameters(), lr=configs.optim.lr)

    os.makedirs(f"{args.output_dir}/{args.exp_name}", exist_ok=True)
    for episode, rewards in reinforce_multiagent_with_phi(
                    env,
                    policy_nets,
                    phi_net,
                    policy_optimizers,
                    phi_optimizer,
                    configs.optim.num_episodes,
                    configs.optim.steps_per_episode,
                    configs.optim.gamma,
                    configs.optim.lambda_phi,
                    print_every=25,
                    reward_aggregation=configs.optim.reward_aggregation,
                    actionsLiteral=configs.env.actionsLiteral):
        torch.save(
            {
                "configs": configs,
                "rewards": rewards,
                "ckpt": [net.state_dict() for net in policy_nets]
            }, f"{args.output_dir}/{args.exp_name}/episode_{episode}.pt")


if __name__ == "__main__":
    main()