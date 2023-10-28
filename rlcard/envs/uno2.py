import numpy as np
from collections import OrderedDict

from rlcard.envs import Env
from rlcard.games.uno2 import Game
from rlcard.games.uno2.utils import encode_hand, encode_target, encode_war_stack, encode_enemy_hand_size
from rlcard.games.uno2.utils import ACTION_SPACE, ACTION_LIST
from rlcard.games.uno2.utils import cards2list

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        'starting_cards': 7,
        'allow_step_back': False,
        'seed': None
        }

class Uno2Env(Env):

    def __init__(self, config):
        self.name = 'uno2'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.state_shape = [[4, 4, 15] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        # print('\nState: ', state)
        obs = np.zeros((4, 4, 15), dtype=float)
        encode_hand(obs[:2], state['hand'])
        encode_target(obs[2], state['target'])
        obs[3][0][0] = min(state['enemy_hand_size'] / 15, 1)
        if state['war_stack_size'] > 0:
            obs[3][0][1] = 1
        obs[3][0][2] = min(state['war_stack_size'] / 15, 1)
        if state['played_wild_4']:
            obs[3][0][3] = 1
        obs[3][0][4] = min(len(state['hand']) / 15, 1)
        legal_action_id = self._get_legal_actions()
        extracted_state = {'obs': obs, 'legal_actions': legal_action_id}
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def get_payoffs(self):
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        legal_ids = self._get_legal_actions()
        if action_id in legal_ids:
            return ACTION_LIST[action_id]
        return ACTION_LIST[np.random.choice(legal_ids)]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = {ACTION_SPACE[action]: None for action in legal_actions}
        return OrderedDict(legal_ids)

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['num_players'] = self.num_players
        state['hand_cards'] = [cards2list(player.hand)
                               for player in self.game.players]
        state['played_cards'] = cards2list(self.game.round.played_cards)
        state['target'] = self.game.round.target.str
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.round.get_legal_actions(
            self.game.players, state['current_player'])
        state['war_stack_size'] = self.game.dealer.war_stack_size
        return state
