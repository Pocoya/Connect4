import gym
from gym import spaces
import numpy as np

class Connect4Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, rows=6, cols=7, win_length=4):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.win_length = win_length

        self.action_space = spaces.Discrete(self.cols)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.rows, self.cols), dtype=np.int8)

        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1 # Player 1 starts
    
    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        return self._get_observation()
    

    def step(self, action):
        """
        Action for the self.current_player
        Returns:
            observation (np.array): The board state from the perspective if the next player.
            reward (float): Reward for the action taken by self.current_player
            done (bool): Whether the game has ended
            info (dict): Additional information (for debugging)
        """
        if not self._is_valid_action(action):
            obs = self._get_observation()
            reward = -10.0
            done = True
            info = {"error": "Invalid move by player " + str(self.current_player)}
            return obs, reward, done, info
        
        # Place the piece
        row_placed = self._place_piece(action, self.current_player)
        
        if self._check_win_from_move(row_placed, action, self.current_player):
            next_obs = self._get_observation()
            reward = 1.0
            done = True
            return next_obs, reward, done, {}
        
        if self._is_draw():
            next_obs = self._get_observation()
            reward = 0.0
            done = True
            return next_obs, reward, done, {}
        
        self.current_player *= -1
        next_obs = self._get_observation()
        reward = 0.0
        done = False
        return next_obs, reward, done, {}
    

    def render(self, mode="human"):
        """ Basic text-based rendering """
        print("  " + " ".join(map(str, range(self.cols))))
        for r in range(self.rows):
            row_str = " ".join(
                "X" if cell == 1 else "O" if cell == -1 else "."
                for cell in self.board[r]
            )
            print(f"{r} {row_str}")
        print("-" * (self.cols * 2 + 2))
        print(f"Current Player: {'X' if self.current_player == 1 else 'O'}")

    def _get_observation(self):
        return self.board.copy() * self.current_player

    def get_valid_actions(self):
        """List of valid actions, enough if top cells empty """
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def _is_valid_action(self, col):
        if not (0 <= col < self.cols):
            return False
        return self.board[0, col] == 0 # top cell empty?
    
    def _place_piece(self, col, player):
        """Place piece, return row of the placed piece or None if full"""
        for r in range(self.rows - 1, -1, -1): # (start, stop (excl.), decr.)
            if self.board[r, col] == 0:
                self.board[r, col] = player
                return r
        return None
    
    def _is_draw(self):
        """Check if board is full"""
        return np.all(self.board[0, :] != 0) # true if top row is full
    
    def _check_win_from_move(self, row, col, player):
        """Checks if the move at (row, col) by 'player' results in a win"""
        if row is None:
            return False
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # one direction
            for i in range(1, self.win_length):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            # opposite direction
            for i in range(1, self.win_length):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break
        
            if count >= self.win_length:
                return True
        return False

