''' A toy example of playing against rule-based bot on UNO
'''

import os
import argparse

import rlcard
from rlcard import models
from rlcard.agents.human_agents.uno_human_agent import HumanAgent, _print_action
from rlcard.agents import NFSPAgent, DQNAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
)

parser = argparse.ArgumentParser("DQN/NFSP human play in RLCard")
parser.add_argument(
    '--load_model_dir',
    type=str,
    default=None,
)
args = parser.parse_args()

env = rlcard.make('uno2')
human_agent = HumanAgent(env.num_actions)

device = get_device()
# if model path includes 'dqn', then use DQNAgent, otherwise use NFSPAgent
if 'dqn' in args.load_model_dir:
    rl_agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
            save_every=1000000,
        )
else:
    rl_agent = NFSPAgent()

rl_agent.load(args.load_model_dir)

env.set_agents([
    human_agent,
    rl_agent,
])

env.debug = True

print(">> UNO rule model V1")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        # if action_record[-i][0] == state['current_player']:
        #     break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses ', end='')
        _print_action(pair[1])
        print('')

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win!')
    else:
        print('You lose!')
    print('')
    input("Press any key to continue...")
