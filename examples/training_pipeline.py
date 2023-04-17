import os
import argparse
import yaml

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model; assuming checkpoints
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
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
    if config['agent']['algorithm'] == 'dqn':
        from rlcard.agents import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=config['agent']['mlp_layers'],
            device=device,
        )
        
    print_current_episode = config['pipeline']['print_current_episode_info']

    # Iterate through the pipeline stages
    for stage in config['stages']:
        print(f'Running stage: {stage["name"]}')
        current_config = defaults.copy()
        current_config.update(stage)
        # Set adverse agents
        adverse_agents = []
        for i, adv_agent in enumerate(stage['adverse_agents']):
            agent = load_model(adv_agent['model_path'], env=env, device=device, position=i + 1)
            adverse_agents.append(load_model(adv_agent['model_path'], env, device))
            
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
                    logger.log_performance(
                        episode,
                        tournament(
                            env,
                            current_config['num_eval_games'],
                        )[0]
                    )

                # Save the model
                if episode % current_config['save_every'] == 0:
                    save_path = os.path.join(stage['log_dir'], f'model_{episode}.pth')
                    torch.save(agent, save_path)
                    print(f'Model saved in {save_path}')

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