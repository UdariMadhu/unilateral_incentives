# Unilateral Incentives

This code is for learning a zero-determinant strategies in two-player multi-agent reinforcement learning games.

The code includes following modules:
- `configs.yaml`: Configs for all experiments
- `policy_gradient.py`: Train both agents using the baseline policy gradient algorithm
- `policy_gradient_with_enforcer.py`: Train one agent with proposed zero-determinant strategy and the other agent as a selfish policy gradient learner.
- `utils.py`: Includes the learning algorithms for baseline and proposed method.

First install the requirements with `pip install -r requirements.txt`

The you can run the policy gradient with zero-determinant strategy simply with following command:
`python policy_gradient_with_enforcer.py --exp_name pg_zero_det`
