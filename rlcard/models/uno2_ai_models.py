''' UNO rule models
'''

import numpy as np
from itertools import permutations

import rlcard
from rlcard.models.model import Model
import torch
import torch.nn as nn
import json
import os

ROOT_PATH = rlcard.__path__[0]
action_space_path = os.path.join(ROOT_PATH, 'games/uno2/jsondata/action_space.json')
action_space = json.load(open(action_space_path, 'r'))
ai_action_space = {k: v for k, v in action_space.items() if v < 60}

class UnoNN(nn.Module):
    def __init__(self):
        super(UnoNN, self).__init__()
        self.fc1 = nn.Linear(120, 256) # 60 (top card) + 60 (hand)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 60)  # Output: 60 (picked card)

    def forward(self, top_card, hand):
        x = torch.cat((top_card, hand), dim=0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class UnoCard:
    def __init__(self, color, figure):
        self.figure = figure
        self.color = color

    def from_string(card_str):
        color, figure = card_str.split('-')
        return UnoCard(color, figure)

    def __repr__(self):
        return f"Card({self.color}, {self.figure})"

    def __str__(self):
        return f"{self.color}-{self.figure}"

    def card_id(self):
        return ai_action_space[self.__str__()]

    def to_vector(self):
        # return a one-hot vector of the card corresponding to the id
        card_id = self.card_id()
        card_vector = [0] * len(ai_action_space)
        card_vector[card_id] = 1
        # turn into tensor
        return torch.tensor(card_vector,  dtype=torch.float)

    def id_for_hand(self):
        return self.card_id()

class UnoHand:
    def __init__(self, cards):
        self.cards = cards

    def from_array(card_array):
        cards = []
        for card_str in card_array:
            card = UnoCard.from_string(card_str)
            cards.append(card)
        return UnoHand(cards)

    def to_vector(self):
        # return an n-hot vector of the hand
        # n-hot meaning a vector of 0s/1s/... of length len(action_space)
        hand_vector = [0] * len(ai_action_space)
        for card in self.cards:
            # if card is a wild card, add as wild of each color
            if card.figure == "wild" or card.figure == "wild_draw_four":
                for color in ["r", "g", "b", "y"]:
                    local_card = UnoCard(color, card.figure)
                    card_id = local_card.card_id()
                    hand_vector[card_id] += 1
            else:
                card_id = card.card_id()
                hand_vector[card_id] += 1
        # turn into tensor
        return torch.tensor(hand_vector, dtype=torch.float)

    def __repr__(self):
        return f"Hand({self.cards})"

    def __str__(self):
        return f"{self.cards}"

class UNOAIRuleAgentV1(object):
    ''' UNO Rule agent version 3
    '''

    def __init__(self):
        self.use_raw = True
        self.debug = False
        model_data = torch.load('model-1706454375.pth')['state_dict']
        self.model = UnoNN()
        self.model.load_state_dict(model_data)
        self.model.eval()
        self.card_str_by_id = {v: k for k, v in ai_action_space.items()}


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

        tensor_hand = UnoHand.from_array(hand).to_vector()
        tensor_top_card = UnoCard.from_string(state['target']).to_vector()
        # import pdb; pdb.set_trace()

        with torch.no_grad():
            outputs = self.model(tensor_top_card, tensor_hand)
            card_id = int(torch.argmax(outputs))

        chosen_card = self.card_str_by_id[card_id]
        legal_actions_except_draw_and_pass = [action for action in legal_actions if action not in ['draw', 'pass']]
        input_cards = [state['target']] + hand
        # target_card


        play = False

        # if chain[0] is playable, play it, otherwise draw
        if chosen_card in legal_actions_except_draw_and_pass:
            play = chosen_card
        elif legal_actions_except_draw_and_pass:
            # if there are playable cards, play the first one
            play = legal_actions_except_draw_and_pass[0]
        elif 'pass' in legal_actions_except_draw_and_pass:
            play =  'pass'
        else:
            play = 'draw'

        if self.debug:
            print(f'Target:', state['target'])
            print(f'Input cards: {input_cards}')
            print(f'Hand: {hand}')
            print(f'Chosen card: {chosen_card}')
            print(f'Play: {play}')
            print('\n'*3)


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


class UNOAIRuleModelV1(Model):
    ''' UNO AI/Rule Model i.e. approximates an "optimal" rule based agent
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('uno2')

        rule_agent = UNOAIRuleAgentV1()
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



