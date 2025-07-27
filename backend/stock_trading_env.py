import gym
from gym import spaces
import numpy as np


class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()

        self.df = df.reset_index()
        self.initial_balance = initial_balance

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: [balance, stock_price, shares_held]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        stock_price = float(self.df['Close'].iloc[self.current_step].item())
        return np.array([self.balance, stock_price, self.shares_held], dtype=np.float32)

    def step(self, action):
        stock_price = float(self.df['Close'].iloc[self.current_step].item())

        if action == 1:  # Buy
            if self.balance >= stock_price:
                self.shares_held += 1
                self.balance -= stock_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += stock_price

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        reward = self.balance + self.shares_held * stock_price - self.initial_balance
        obs = self._next_observation()

        return obs, reward, done, {}

    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares Held: {self.shares_held}")