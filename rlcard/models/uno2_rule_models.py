''' UNO rule models
'''

import numpy as np
from itertools import permutations

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




class UNORuleAgentV3(object):
    ''' UNO Rule agent version 3
    '''

    def __init__(self):
        self.use_raw = True
        self.debug = False

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


        legal_actions_except_draw_and_pass = [action for action in legal_actions if action not in ['draw', 'pass']]
        # import pdb; pdb.set_trace()
        input_cards = [state['target']] + hand
        chain = []
        if legal_actions_except_draw_and_pass:
            if len(input_cards) >= 10:
                # take 10 random cards from hand
                # keepp only ONE of each wild from legal_actions_except_draw_and_pass
                # first select legal actions except wilds
                corrected_legal_actions_except_draw_and_pass = [action for action in legal_actions_except_draw_and_pass if action[2:6] != 'wild']
                # identify the first wild
                wilds = [action for action in legal_actions_except_draw_and_pass if action[2:] == 'wild']
                first_wild = [wilds[0]] if len(wilds) > 0 else []
                # identify the first wild draw 4
                wild_draw_4s = [action for action in legal_actions_except_draw_and_pass if action[2:] == 'wild_draw_4']
                first_wild_draw_4 = [wild_draw_4s[0]] if len(wild_draw_4s) > 0 else []
                # add the first wild and first wild draw 4 to corrected_legal_actions_except_draw_and_pass
                corrected_legal_actions_except_draw_and_pass = corrected_legal_actions_except_draw_and_pass + first_wild + first_wild_draw_4
                input_cards = [state['target']] + corrected_legal_actions_except_draw_and_pass[0:9]
            chain = self.find_highest_probability_path(input_cards)

        play = False

        # if chain[0] is playable, play it, otherwise draw
        if len(chain) > 0 and chain[0] in legal_actions_except_draw_and_pass:
            if (self.figure(chain[0]) == 'wild' or self.figure(chain[0]) == 'wild_draw_4') and len(chain) > 1:
                play = self.color(chain[1]) + '-' + self.figure(chain[0])
            else:
                play = chain[0]
        elif 'pass' in legal_actions_except_draw_and_pass:
            play =  'pass'
        else:
            play = 'draw'

        if self.debug:
            print(f'Target:', state['target'])
            print(f'Input cards: {input_cards}')
            print(f'Hand: {hand}')
            print(f'Chain: {chain}')
            print(f'Play: {play}')
            print('\n'*3)

        #import pdb; pdb.set_trace()

        return play



    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []

    def figure(self, card):
        ''' Return the figure of the card

        Args:
            card (str): A UNO card string

        Returns:
            figure (str): The figure of the card
        '''
        return card[2:]

    def color(self, card):
        ''' Return the color of the card

        Args:
            card (str): A UNO card string

        Returns:
            color (str): The color of the card
        '''
        return card[0]

    def transition_probability(self, card_from, card_to, i):
        if self.figure(card_from) == 'skip' and (self.color(card_to) == self.color(card_from) or self.figure(card_to) == 'skip'):
            return 1.00
        elif self.figure(card_to) == 'wild_draw_4':
            return 1.00 #0.95
        elif self.figure(card_to) == 'wild':
            return 0.95 #0.90
        elif self.figure(card_from) in ['wild', 'wild_draw_4'] and i != 0:
            return 0.85
        elif self.figure(card_from) == self.figure(card_to) and self.color(card_from) == self.color(card_to):
            return 0.85
        elif self.color(card_from) == self.color(card_to):
            return 0.60
        elif self.figure(card_from) == self.figure(card_to):
            return 0.20
        else:
            return 0.05

    def discount_probability(self, prob_list):
        discounted_probs = []
        discount_factor = 1.0
        for prob in reversed(prob_list):
            discounted_prob = prob * discount_factor
            if self.debug:
                print(f'Prob: {prob}, Discount factor: {discount_factor}, Discounted prob: {discounted_prob}')
            discounted_probs.insert(0, discounted_prob)
            discount_factor *= 0.99  # Apply a discounting factor for transitions
        return discounted_probs

    def calculate_path_probability(self, path):
        probs = [self.transition_probability(path[i], path[i+1], i) for i in range(len(path)-1)]
        discounted_probs = self.discount_probability(probs)
        total_prob = 1
        for prob in discounted_probs:
            total_prob += prob
        return total_prob

    def find_highest_probability_path(self, cards):
        # print(f'Cards: {cards}')
        start_card = cards[0]  # Assume the first card is fixed
        if self.figure(start_card) == 'wild' or self.figure(start_card) == 'wild_draw_4':
            start_card = self.color(start_card) + '-0'
        remaining_cards = cards[1:]  # Exclude the first card from the permutations


        highest_prob = 0
        best_path = []
        for perm in permutations(remaining_cards):
            candidate_path = [start_card] + list(perm)
            current_prob = self.calculate_path_probability(candidate_path)
            if current_prob > highest_prob:
                highest_prob = current_prob
                best_path = perm
        return list(best_path)

class UNORuleModelV3(Model):
    ''' UNO Rule Model version 2
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('uno2')

        rule_agent = UNORuleAgentV3()
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



