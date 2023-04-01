import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make("leduc-holdem")
print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)


agent = RandomAgent(num_actions=env.num_actions)
env.set_agents([agent for _ in range(env.num_players)])

trajectories, player_wins = env.run(is_training=False)

print(trajectories)

print(player_wins)
