from copy import deepcopy
import numpy as np

from rlcard.games.uno2 import Dealer
from rlcard.games.uno2 import Player
from rlcard.games.uno2 import Round


class UnoGame:
    def __init__(self, allow_step_back=False, num_players=2, starting_cards=7):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        self.payoffs = [0 for _ in range(self.num_players)]
        self.starting_cards = starting_cards

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players
        '''
        self.num_players = game_config['game_num_players']

    def init_game(self, cards = 7):
        ''' Initialize players and state

        Returns:
            (tuple): Tuple containing:

                (dict): The first state in one game
                (int): Current player's id
        '''
        self.starting_cards = cards
        # Initalize payoffs
        self.payoffs = [0 for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)
        self.dealer.starting_cards = 7

        # Initialize four players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        # Deal 7 cards to each player to prepare for the game
        for player in self.players:
            self.dealer.deal_cards(player, cards)

        # Initialize a Round
        self.round = Round(self.dealer, self.num_players, self.np_random)

        # flip and perfrom top card
        top_card = self.round.flip_top_card()
        self.round.perform_top_card(self.players, top_card)

        # Save the hisory for stepping back to the last state.
        self.history = []

        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''

        if self.allow_step_back:
            # First snapshot the current state
            his_dealer = deepcopy(self.dealer)
            his_round = deepcopy(self.round)
            his_players = deepcopy(self.players)
            self.history.append((his_dealer, his_players, his_round))

        self.round.proceed_round(self.players, action)
        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if not self.history:
            return False
        self.dealer, self.players, self.round = self.history.pop()
        return True

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        state = self.round.get_state(self.players, player_id)
        state['num_players'] = self.get_num_players()
        state['current_player'] = self.round.current_player
        state['war_stack_size'] = self.dealer.war_stack_size
        state['played_wild_4'] = self.dealer.played_wild_4
        state['enemy_hand_size'] = len(self.players[1 - player_id].hand) # TODO: generalize
        return state

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        winner = self.round.winner
        if winner is not None and len(winner) == 1:
            self.payoffs[winner[0]] = 1 - self.turn / 1000
            self.payoffs[1 - winner[0]] = -1
        else:
            # assign the payoff -1 to all players
            self.payoffs = [-1 for _ in range(self.num_players)]
        return self.payoffs

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''

        return self.round.get_legal_actions(self.players, self.round.current_player)

    def get_num_players(self):
        ''' Return the number of players in Limit Texas Hold'em

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    @staticmethod
    def get_num_actions():
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 61 actions
        '''
        return 62

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.round.current_player

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        return self.round.is_over
