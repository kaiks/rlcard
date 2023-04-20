import os
import argparse
import yaml

import torch

import rlcard
from rlcard.agents import RandomAgent, DQNAgent, NFSPAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def load_agent_from_checkpoint(checkpoint_data):
    agent_class = getattr(rlcard.agents, checkpoint_data['agent_type'])
    return agent_class.from_checkpoint(checkpoint_data)

def load_model(model, env, position=None, device=None):
    if os.path.isfile(model):  # Torch model; assuming checkpoints
        agent_data = torch.load(model, map_location=device)
        if isinstance(agent_data, dict) and 'agent_type' in agent_data:
            print(f"Loading {agent_data['agent_type']} agent from checkpoint.")
            agent = load_agent_from_checkpoint(agent_data)
        else:
            print("No agent type identified. Assuming we can load it from the model path using just torch.")
            agent = agent_data
        agent.set_device(device)
    elif os.path.isdir(model):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model)
        agent.load()
    elif model == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model).agents[position]
    
    return agent


defaults = {
    'num_episodes': 100000,
    'evaluate_every': 1000,
    'num_eval_games': 100,
}

def train_pipeline(pipeline_config):
    # Load pipeline configuration from the YAML file
    with open(pipeline_config, 'r') as file:
        config = yaml.safe_load(file)

    # Check whether GPU is available
    device = get_device()

    # Seed numpy, torch, random
    if 'seed' in config['env']:
        set_seed(config['env']['seed'])
    else:
        print('No seed is set. Random seed will be used.')
        set_seed(seed = None)

    # Make the environment
    env = rlcard.make(config['env']['name'])

    # Initialize the agent    
    if 'checkpoint_path' in config['agent']:
        print(f"Loading agent from checkpoint: {config['agent']['checkpoint_path']}")
        print(f"Configuration parameters will be ignored.")
        agent = load_model(config['agent']['checkpoint_path'], env, device=device)
    elif config['agent']['algorithm'] == 'dqn':
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            device=device,
            **config['agent']['parameters']
        )
    print_current_episode = config['pipeline']['print_current_episode_info']

    # Iterate through the pipeline stages
    for stage in config['stages']:
        best_reward = -99999999
        print(f'Running stage: {stage["name"]}')
        current_config = defaults.copy()
        current_config.update(stage)
        # Set adverse agents
        adverse_agents = []
        for i, adv_agent in enumerate(stage['adverse_agents']):
            adv_agent = load_model(adv_agent['algorithm'], env=env, device=device, position=i + 1)
            adverse_agents.append(adv_agent)
            
        if 'agent_parameters' in stage:
            for key, value in stage['agent_parameters'].items():
                setattr(agent, key, value)

        # Set the environment parameters and agents
        env.set_agents([agent] + adverse_agents)
        env.set_game_config(stage['env'])

        # Start training
        with Logger(current_config['log_dir']) as logger:
            for episode in range(current_config['num_episodes']):
                if print_current_episode:
                    print(f'Current episode: {episode}')
                
                # Generate data from the environment
                trajectories, payoffs = env.run(is_training=True)

                # Reorganize the data to be state, action, reward, next_state, done
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agent memory, and train the agent
                for ts in trajectories[0]:
                    agent.feed(ts)

                # Evaluate the performance. Play with random agents.
                if episode % current_config['evaluate_every'] == 0:
                    tournament_payoffs = tournament(env, current_config['num_eval_games'])
                    logger.log_performance(episode,tournament_payoffs[0])
                    if tournament_payoffs[0] > best_reward:
                        best_reward = tournament_payoffs[0]
                        agent.save_checkpoint(current_config['log_dir'], 'best_model.pt')
                        print('Model saved in episode #{} with reward {}'.format(episode, best_reward))

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, config['agent']['algorithm'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard with pipeline")
    parser.add_argument(
        '--pipeline_config',
        type=str,
        default='pipeline_config.yaml',
        help='Path to the pipeline configuration YAML file',
    )

    args = parser.parse_args()
    train_pipeline(args.pipeline_config)