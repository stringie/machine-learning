
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')

import gym
import matplotlib
import numpy as np
import sys
from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")
from BlackjackEnv import BlackjackEnv
from cliffwalking import CliffWalkingEnv
from windyGrid import WindyGridworldEnv
import plotting

matplotlib.style.use('ggplot')


# In[2]:


env = BlackjackEnv()


# In[3]:


def mc(env, nepisodes, gamma=1.0, epsilon=0.1):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    def policy_fn(observation):
        A = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
        
    policy = policy_fn
        
    #Number of episodes for evaluation limit
    for episode_i in range(1, nepisodes + 1):
        
        #Generate episode
        episode = []
        state = env.reset()
        finished = False
        while not finished:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            s_prime, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                finished = True
            state = s_prime
        
        #for each state record its state value
        sa_pairs = set([(x[0] , x[1]) for x in episode])
        for state, action in sa_pairs:
            sa_pair = (state, action)
            first_occ_indx = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
            G = sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occ_indx:])])
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

    return Q, policy


# In[7]:


Q, policy = mc(env, 100000)


# In[10]:


V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal V")


# In[11]:


V

