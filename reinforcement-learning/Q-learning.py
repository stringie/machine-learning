
# coding: utf-8

# In[1]:


import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
    sys.path.append("../")
    
from collections import defaultdict
from cliffwalking import CliffWalkingEnv
from BlackjackEnv import BlackjackEnv
import plotting

matplotlib.style.use('ggplot')


# In[2]:


env = BlackjackEnv()


# In[3]:


def q_learning(env, nepisodes, gamma=1.0, alpha=0.5):
    epsilon = 0.1
    global_time_step = 1
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(nepisodes),
        episode_rewards=np.zeros(nepisodes))    
    
    def policy_fn(observation):
        A = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    
    # The policy we're following
    policy = policy_fn
    
    for i_episode in range(nepisodes): 
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            epsilon = 1.0/global_time_step
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            global_time_step += 1    
            
            if done:
                break
                
            state = next_state
    
    return Q, stats


# In[4]:


Q, stats = q_learning(env, 50000)


# In[5]:


plotting.plot_episode_stats(stats)

