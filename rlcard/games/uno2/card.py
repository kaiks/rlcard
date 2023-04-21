from termcolor import colored

class UnoCard:

    info = {'type':  ['number', 'action', 'wild'],
            'color': ['r', 'g', 'b', 'y'],
            'trait': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'skip', 'reverse', 'draw_2', 'wild', 'wild_draw_4']
            }

    def __init__(self, card_type, color, trait):
        ''' Initialize the class of UnoCard

        Args:
            card_type (str): The type of card
            color (str): The color of card
            trait (str): The trait of card
        '''
        self.type = card_type
        self.color = color
        self.trait = trait
        self.str = self.get_str()
        
    @classmethod
    def init_with_str(cls, card_str):
        ''' Initialize the class of UnoCard with a string

        Args:
            card_str (str): The string of card

        Return:
            (UnoCard): The UnoCard object
        '''
        color, trait = card_str.split('-')
        if trait in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            card_type = 'number'
        elif trait in ['skip', 'reverse', 'draw_2']:
            card_type = 'action'
        else:
            card_type = 'wild'

        return cls(card_type, color, trait)

    def get_str(self):
        ''' Get the string representation of card

        Return:
            (str): The string of card's color and trait
        '''
        return self.color + '-' + self.trait
    
    def is_war_playable(self):
        ''' Check if the card is playable in War
            A card is playable in War if it is a draw_2, wild_draw_4, or reverse card

        Return:
            (bool): True if the card is playable
        '''
        return self.trait == 'draw_2' or self.trait == 'wild_draw_4' or self.trait == 'reverse'
    
    def score(self):
        ''' Get the score of the card

        Return:
            (int): The score of the card
        '''
        if self.trait == 'wild_draw_4':
            return 50
        elif self.trait == 'draw_2':
            return 20
        elif self.trait == 'skip':
            return 20
        elif self.trait == 'reverse':
            return 20
        elif self.trait == 'wild':
            return 50
        else:
            return int(self.trait)


    @staticmethod
    def print_cards(cards, wild_color=False):
        ''' Print out card in a nice form

        Args:
            card (str or list): The string form or a list of a UNO card
            wild_color (boolean): True if assign color to wild cards
        '''
        if isinstance(cards, str):
            cards = [cards]
        for i, card in enumerate(cards):
            if card == 'draw' or card == 'pass':
                trait = card.capitalize()
            else:
                color, trait = card.split('-')
                if trait == 'skip':
                    trait = 'Skip'
                elif trait == 'reverse':
                    trait = 'Reverse'
                elif trait == 'draw_2':
                    trait = 'Draw-2'
                elif trait == 'wild':
                    trait = 'Wild'
                elif trait == 'wild_draw_4':
                    trait = 'Wild-Draw-4'

            if trait == 'Draw' or trait == 'Pass' or (trait[:4] == 'Wild' and not wild_color):
                print(trait, end='')
            elif color == 'r':
                print(colored(trait, 'red'), end='')
            elif color == 'g':
                print(colored(trait, 'green'), end='')
            elif color == 'b':
                print(colored(trait, 'blue'), end='')
            elif color == 'y':
                print(colored(trait, 'yellow'), end='')

            if i < len(cards) - 1:
                print(', ', end='')
