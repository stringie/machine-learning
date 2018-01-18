#%%
import gym
import numpy as np

env = gym.make('Breakout-v0')

s = env.reset()
# s = np.stack([s]*4, axis=2)
#%%
print(np.expand_dims(s, 2).shape)
# print(np.append(s[:, :, 1, :], np.expand_dims(s, 2), axis=2 )