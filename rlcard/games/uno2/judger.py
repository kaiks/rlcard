import numpy as np
class UnoJudger:

    @staticmethod
    def judge_winner(players, np_random):
        ''' Judge the winner of the game

        Args:
            players (list): The list of players who play the game

        Returns:
            (list): The player id of the winner
        '''
        self.np_random = np_random
        count = [len(p.hand) for p in players]
        return np.argwhere(count == np.min(count)).flatten().tolist()
