''' UNO rule models
'''

import numpy as np

import rlcard
from rlcard.models.model import Model

class UNORuleAgentV2(object):
    ''' UNO Rule agent version 2
    '''

    def __init__(self):
        self.use_raw = True

    def step(self, state):
        ''' Predict the action given raw state. Play the card that gives the longest chain of playable cards.

        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''

        legal_actions = state['raw_legal_actions']
        state = state['raw_obs']
        hand = state['hand']

        #print(f'Legal actions: {legal_actions}')

        legal_actions_except_draw_and_pass = [action for action in legal_actions if action not in ['draw', 'pass']]
        chain = self.longest_chain(hand, legal_actions_except_draw_and_pass, state['target'])
        # print hand and chain
        #print(f'Hand: {hand}')
        #print(f'Chain: {chain}')

        if chain:
            return chain[0]
        else:
            return np.random.choice(legal_actions)



    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []

    def longest_chain(self, hand, starting_legal_actions, top_card):
        ''' Find the longest chain of playable cards in hand

        Args:
            hand (list): A list of UNO card string
            starting_legal_actions (list): A list of UNO card string playable currently, except 'draw' and 'pass'
            top_card (str): A UNO card string

        Returns:
            chain (list): A list of UNO card string
        '''
        max_length = -1
        longest_path = []
        # print(f'Top card: {top_card}')
        # print(f'Starting legal actions: {starting_legal_actions}')
        for start_vertex in starting_legal_actions:
            # print(f'Vertex: {start_vertex}')
            visited = {vertex: False for vertex in hand}
            path = [start_vertex]
            stack = [start_vertex]
            while stack:
                curr_vertex = stack[-1]
                visited[curr_vertex] = True
                neighbors = [neighbor for neighbor in hand if self.is_playable(neighbor, curr_vertex) and not visited[neighbor]]
                if not neighbors:
                    path_length = len(path)
                    if path_length > max_length:
                        max_length = path_length
                        # copy path to longest_path
                        longest_path = path[:]
                    stack.pop()
                    path.pop()
                else:
                    next_vertex = neighbors[0]
                    stack.append(next_vertex)
                    path.append(next_vertex)
        return longest_path

    def is_playable(self, card, top_card):
        ''' Check if card is playable on top_card

        Args:
            card (str): A UNO card string
            top_card (str): A UNO card string

        Returns:
            playable (boolean): True if card is playable on top_card
        '''
        is_playable = card[0] == top_card[0] or (card[2] == top_card[2] and not card[2] == 'w') or card[2:6] == 'wild'
        # print(f'Comparison components: {card[0]} == {top_card[0]} or {card[2]} == {top_card[2]} or {card[2:6]} == "wild"')
        # print(f'Card: {card}, Top card: {top_card}, Playable: {is_playable}')

        return is_playable

class UNORuleModelV2(Model):
    ''' UNO Rule Model version 2
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('uno2')

        rule_agent = UNORuleAgentV2()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.rule_agents

    @property
    def use_raw(self):
        ''' Indicate whether use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return True



