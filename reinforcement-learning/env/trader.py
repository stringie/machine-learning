
# coding: utf-8

# In[123]:


import gym
import pyscreenshot as psc
import pytesseract as pt
import pyautogui as pag
import threading
import time
import matplotlib.pyplot as plt
from collections import deque
from gym import spaces
from gym.utils import seeding


# In[156]:


class TraderEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.nA = 3
        self.buys = deque()
        self.sells = deque()
        self.history = deque()
        self.reset()
        
    def reset(self):
        self.state = self.stock()
        self.history.append(self.state)
        if len(self.history) > 100:
            self.history.popleft()
        
        
    def step(self, action):
        reward = 0
        done = False
        
        if action == 0:
            self.wait()
        elif action == 1:
            reward = self.buy()
        elif action == 2:
            reward = self.sell()
    
        return self.getObs(), reward, done, {}
            
    def getObs(self):
        return (self.state, list(self.history), list(self.buys), list(self.sells))
    
    def click(self, x, y):
        pag.moveTo(x, y)
        pag.click()
        
    def calculateBuyReward(self):
        if len(self.sells) == 0:
            self.buys.append(self.state)
            return 0
        sell = self.sells.popleft()
        return sell - self.state
        
    def calculateSellReward(self):
        if len(self.buys) == 0:
            self.sells.append(self.state)
            return 0
        buy = self.buys.popleft()
        return self.state - buy
        
    def wait(self):
        self.reset()
        #plt.plot(range(1, len(self.history) + 1), list(self.history))
        #plt.show()
        
    def buy(self):
        x = 635
        y = 663
        self.click(x, y)
        self.reset()
        return self.calculateBuyReward()
        
    def sell(self):
        x = 326
        y = 663
        self.click(x, y)
        self.reset()
        return self.calculateSellReward()
        
    def stock(self):
        im = psc.grab(bbox=(100,164,207,188))
        price = float(pt.image_to_string(im).replace(" ", ""))
        return price

