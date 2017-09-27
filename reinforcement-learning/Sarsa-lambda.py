
# coding: utf-8

# In[3]:


import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
    sys.path.append("../")
    
from collections import defaultdict
from windyGrid import WindyGridworldEnv
from cliffwalking import CliffWalkingEnv
import plotting

matplotlib.style.use('ggplot')


# In[2]:


env = CliffWalkingEnv()


# In[3]:


#lmbd = lambda parameter
#do not go above 0.8
def sarsa_lambda(env, nepisodes, gamma=1.0, alpha=0.5, lmbd=0.8):
    epsilon = 0.1
    global_time_step = 1
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    E = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for state in range(env.nS):
        for action in range(env.nA):
            E[state][action] = 0.0
            
    
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(nepisodes),
        episode_rewards=np.zeros(nepisodes))
    
    def policy_fn(observation):
        A = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    
    policy = policy_fn
    
    for i_episode in range(nepisodes):
        
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        for t in itertools.count():
            epsilon *= 0.99
            s_prime, reward, done, _ = env.step(action)
            
            a_prime_probs = policy(s_prime)
            a_prime = np.random.choice(np.arange(len(a_prime_probs)), p=a_prime_probs)
            
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            delta = reward + gamma * Q[s_prime][a_prime] - Q[state][action]
            E[state][action] += 1
            
            for state in range(env.nS):
                for action in range(env.action_space.n):
                    Q[state][action] += alpha * delta * E[state][action]
                    E[state][action] *= gamma * lmbd
            
            if done:
                break
                
            global_time_step += 1
            action = a_prime
            state = s_prime
            
    return Q, stats   


# In[28]:


Q, stats = sarsa_lambda(env, 90)


# In[29]:


plotting.plot_episode_stats(stats)

